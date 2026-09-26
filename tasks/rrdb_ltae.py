import torch
import torch.nn.functional as F
from models.misr_module import RRDBLtaeNet
from utils.hparams import hparams
from trainer import Trainer
import json
import os
from losses.srdiff_loss import (
    pixel_wise_closest_sr_sits_aer_loss,
    grad_pixel_wise_closest_sr_sits_aer_loss,
    temp_gradient_magnitude_consistency_loss,
    gray_value_consistency_loss,
)

from models.sits_aerial_seg_model import SITSAerialSegmenter
from utils.utils import load_ckpt
from losses.focal_smooth import FocalLossWithSmoothing


class RRDBLtae(Trainer):
    def build_model(self):
        self.criterion_aer = FocalLossWithSmoothing(
            hparams["inputs"]["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )
        self.criterion_sat = FocalLossWithSmoothing(
            hparams["inputs"]["num_classes"], gamma=2, alpha=1, lb_smooth=0.2
        )
        self.loss_aux_sat_weight = hparams["hyperparams"]["loss_aux_sat_weight"]
        self.loss_main_sat_weight = hparams["hyperparams"]["loss_main_sat_weight"]

        with open("./tasks/config_rrdb_misr.json", "r") as read_file:
            self.config = json.load(read_file)

        self.sr_model = RRDBLtaeNet(self.config)

        if hparams["cond_net_ckpt"] != "" and os.path.exists(hparams["cond_net_ckpt"]):
            print(
                f"Loading conditioning network checkpoint: {hparams['cond_net_ckpt']}"
            )
            load_ckpt(self.sr_model, hparams["cond_net_ckpt"])
            print("Conditioning network checkpoint loaded successfully.")
        else:
            print(
                f"Conditioning network checkpoint not found or not specified: "
                f"{hparams['cond_net_ckpt']}"
            )

        self.model = SITSAerialSegmenter(sr_model=self.sr_model, config=hparams)
        if hparams["infer"]:
            if hparams["sr_net_ckpt"] != "" and os.path.exists(hparams["sr_net_ckpt"]):
                load_ckpt(self.model, hparams["sr_net_ckpt"])
        self.global_step = 0
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams["lr"])

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, 100000, 0.5)

    def closest_lr_sits_aer(self, img_lr_up, closest_indices):
        B, T, C, H, W = img_lr_up.shape
        closest_indices = closest_indices.to(torch.long)

        # Ensure closest_indices is a tensor
        if not torch.is_tensor(closest_indices):
            closest_indices = torch.tensor(closest_indices, device=img_lr_up.device)

        # Gather closest satellite images
        closest_sat_image = img_lr_up[
            torch.arange(B), closest_indices
        ]  # shape (B, C, H, W)
        return closest_sat_image

    def training_step(self, sample):
        # Training Step of HighResLTAE Model
        img_hr = sample["img_hr"]
        img_hr_down = sample["img_hr_down"]
        img_lr = sample["img_lr"]
        labels = sample["labels"]  # torch.Size([4, 2, 3, 160, 160])
        labels_sr = sample["labels_sr"]  # torch.Size([4, 2, 3, 160, 160])
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]
        sc_img_hr = img_hr_down[:, :4, :, :]

        img_sr = self.sr_model(img_lr, dates, self.config)

        pred_img = self.closest_lr_sits_aer(img_sr, closest_idx)
        
        print("pred_img:", pred_img.shape)
        print("sc_img_hr:", sc_img_hr.shape)

        sr_loss = F.l1_loss(pred_img, sc_img_hr, reduction="mean")

        aux_loss = (
            hparams["px_loss_weight"]
            * pixel_wise_closest_sr_sits_aer_loss(img_sr, sc_img_hr, closest_idx)
            + hparams["grad_px_loss_weight"]
            * grad_pixel_wise_closest_sr_sits_aer_loss(img_sr, sc_img_hr, closest_idx)
            + hparams["temp_grad_mag_loss_weight"]
            * temp_gradient_magnitude_consistency_loss(img_sr)
            + hparams["gray_value_px_loss_weight"]
            * gray_value_consistency_loss(img_sr, img_lr)
        )
        final_loss = hparams["main_loss_weight"] * sr_loss + aux_loss
        losses = {"sr": final_loss}

        # Compute the loss at each time step
        # Only 4 channels are used for the loss computation
        # Because the 5th channel is the NDSM band not available in the
        # low-resolution image time series.

        # for classification branches
        cls_sits, multi_outputs, aer_outputs = self.model(
            img_hr, img_sr, dates
        )

        labels_sr = torch.argmax(labels_sr, dim=1) if labels_sr.ndim == 4 else labels_sr
        labels = torch.argmax(labels, dim=1) if labels.ndim == 4 else labels

        aux_loss1 = self.criterion_sat(multi_outputs[2], labels_sr)
        aux_loss2 = self.criterion_sat(multi_outputs[1], labels_sr)
        aux_loss3 = self.criterion_sat(multi_outputs[0], labels_sr)

        # loss for main SITS classification branch
        loss_main_sat = self.criterion_sat(cls_sits, labels_sr)

        # Total loss for SITS branch
        loss_sat = self.loss_main_sat_weight * loss_main_sat + (
            self.loss_aux_sat_weight * aux_loss1
            + self.loss_aux_sat_weight * aux_loss2
            + self.loss_aux_sat_weight * aux_loss3
        )

        # print("labels:", labels.shape)
        # print("aer_outputs:", aer_outputs.shape)

        # labels: torch.Size([2, 512, 512])
        # aer_outputs: torch.Size([2, 13, 512, 512])

        # Loss for AER branch
        loss_aer = self.criterion_aer(aer_outputs, labels.long())

        # The CE loss for the SITS classification branch is done at 1.6m GSD
        # that combines the loss from the SR-diffusion model and the SITS
        #  segmentation branch

        losses["sr"] = hparams["hyperparams"]["loss_weights_aer_sat"][1] * (
            losses["sr"] + loss_sat
        )

        # The CE loss for the AER classification branch is done at 20cm GSD
        losses["aer"] = hparams["hyperparams"]["loss_weights_aer_sat"][0] * loss_aer

        total_loss = sum(losses.values())
        return losses, total_loss

    def sample_and_test(self, sample):
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img_hr = sample["img_hr"]
        img_hr_down = sample["img_hr_down"]
        img_lr = sample["img_lr"]
        labels = sample["labels"]
        labels_sr = sample["labels_sr"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]
        sc_img_hr = img_hr_down[:, :4, :, :]

        img_sr = self.sr_model(img_lr, dates, self.config)

        pred_img = self.closest_lr_sits_aer(img_sr, closest_idx)
        sr_loss = F.l1_loss(pred_img, sc_img_hr, reduction="mean")

        aux_loss = (
            hparams["px_loss_weight"]
            * pixel_wise_closest_sr_sits_aer_loss(img_sr, sc_img_hr, closest_idx)
            + hparams["grad_px_loss_weight"]
            * grad_pixel_wise_closest_sr_sits_aer_loss(img_sr, sc_img_hr, closest_idx)
            + hparams["temp_grad_mag_loss_weight"]
            * temp_gradient_magnitude_consistency_loss(img_sr)
            + hparams["gray_value_px_loss_weight"]
            * gray_value_consistency_loss(img_sr, img_lr)
        )
        final_loss = hparams["main_loss_weight"] * sr_loss + aux_loss
        losses = {"sr": final_loss}

        # Compute the loss at each time step
        # Expand along Time dimension

        # during sampling, only the aer branch is used
        cls_sits, multi_outputs, aer_outputs = self.model(
            img_hr, img_sr, dates
        )

        proba = torch.softmax(aer_outputs, dim=1)
        preds = torch.argmax(proba, dim=1)

        labels_sr = torch.argmax(labels_sr, dim=1) if labels_sr.ndim == 4 else labels_sr
        labels = torch.argmax(labels, dim=1) if labels.ndim == 4 else labels

        aux_loss1 = self.criterion_sat(multi_outputs[2], labels_sr)
        aux_loss2 = self.criterion_sat(multi_outputs[1], labels_sr)
        aux_loss3 = self.criterion_sat(multi_outputs[0], labels_sr)

        # loss for main SITS classification branch
        loss_main_sat = self.criterion_sat(cls_sits, labels_sr)

        # Total loss for SITS branch
        loss_sat = self.loss_main_sat_weight * loss_main_sat + (
            self.loss_aux_sat_weight * aux_loss1
            + self.loss_aux_sat_weight * aux_loss2
            + self.loss_aux_sat_weight * aux_loss3
        )

        # print("labels:", labels.shape)
        # print("aer_outputs:", aer_outputs.shape)

        # labels: torch.Size([2, 512, 512])
        # aer_outputs: torch.Size([2, 13, 512, 512])

        # Loss for AER branch
        loss_aer = self.criterion_aer(aer_outputs, labels.long())

        # The CE loss for the SITS classification branch is done at 1.6m GSD
        # that combines the loss from the SR-diffusion model and the SITS
        #  segmentation branch

        losses["sr"] = hparams["hyperparams"]["loss_weights_aer_sat"][1] * (
            losses["sr"] + loss_sat
        )

        # The CE loss for the AER classification branch is done at 20cm GSD
        losses["aer"] = hparams["hyperparams"]["loss_weights_aer_sat"][0] * loss_aer

        total_loss = sum(losses.values())

        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                sc_img_hr[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
                preds[b],
                labels[b],
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, preds, ret, total_loss
