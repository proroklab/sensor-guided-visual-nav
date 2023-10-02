import torch
import torchvision
import json
from torch import optim, Tensor
from torch.nn import functional as F
from lightning.pytorch import LightningModule
import matplotlib as mpl
import cv2
import yaml
from collections import OrderedDict
from typing import Optional

mpl.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from src.dataset_util.src.env.env_v2 import Env
from src.training.util.cli import MyLightningCLI
from src.training.dataloaders.dataset_sim2real import Sim2RealDataModule


def load_model(checkpoint_path, model):
    model_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
        "state_dict"
    ]

    # A basic remapping is required
    mapping = {k: v for k, v in zip(model_state.keys(), model.state_dict().keys())}
    mapped_model_state = OrderedDict([(mapping[k], v) for k, v in model_state.items()])
    model.load_state_dict(mapped_model_state, strict=False)
    return model


def kl_div_multivariate(mu_1, mu_2, log_var_1, log_var_2):
    eye = torch.eye(32).unsqueeze(0).repeat(16, 1, 1).to(log_var_1.device)
    Var_1 = eye * log_var_1.type(torch.float64).exp().unsqueeze(2)
    Var_2 = eye * log_var_2.type(torch.float64).exp().unsqueeze(2)
    Var_2_inv = torch.inverse(Var_2)
    mlt_v2_v1 = Var_2_inv @ Var_1
    tr_v2_v1 = mlt_v2_v1.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    log_v2_v1 = (torch.det(Var_2) / (torch.det(Var_1) + 1e-7)).log()
    d_mu2_mu1 = (mu_2 - mu_1).type(torch.float64)
    u2_u1_Var2 = (d_mu2_mu1.unsqueeze(1) @ Var_2_inv @ d_mu2_mu1.unsqueeze(2)).flatten()
    n = mu_1.shape[1]
    return (0.5 * (log_v2_v1 - n + tr_v2_v1 + u2_u1_Var2)).type(mu_1.dtype)


class Sim2RealExperiment(LightningModule):
    def __init__(
        self,
        model_checkpoint: str,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_gamma: Optional[float] = 0.98,
    ) -> None:
        super(Sim2RealExperiment, self).__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

        checkpoint_file = Path(model_checkpoint)
        self.model = load_model(checkpoint_file, model).to(self.device)

        self.model.vae_real.load_state_dict(self.model.vae_sim.state_dict())
        for param in self.model.parameters():
            param.requires_grad = False
        enc_trainable_layers = 2
        for param in self.model.vae_real.encoder.features[
            0 : self.model.vae_real.enc_out_layer - enc_trainable_layers + 3
        ].parameters():
            param.requires_grad = True

    def forward(self, datas) -> Tensor:
        result = {}
        result["sim_dirs"] = self.model(
            {"img_norm": datas["sim"], "sensors_pos": datas["sensors_pos"]},
            vae_type="sim",
        )
        result["sim_recons"] = self.model.recons
        result["sim_mu"] = self.model.mu
        result["sim_log_var"] = self.model.log_var

        result["real_dirs"] = self.model(
            {"img_norm": datas["real"], "sensors_pos": datas["sensors_pos"]},
            vae_type="real",
        )
        result["real_recons"] = self.model.recons
        result["real_mu"] = self.model.mu
        result["real_log_var"] = self.model.log_var

        return result

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        datas, labels = batch

        self.model.train()

        result = self.forward(datas)
        dir_loss = F.mse_loss(result["real_dirs"], result["sim_dirs"])

        vae_loss = self.model.vae_real.loss_function(
            result["real_recons"],
            datas["mask"].flatten(0, 1),
            result["real_mu"],
            result["real_log_var"],
            M_N=1 / (84 * 4 * 84),
        )
        loss = dir_loss  # + vae_loss["loss"]

        self.log_dict(
            {
                "vae_reconstructiont": vae_loss["reconstruction_loss"],
                "vae_loss": vae_loss["loss"],
                "vae_kld": vae_loss["kld"],
                "dir_loss": dir_loss,
                "loss": loss,
            },
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        datas, labels = batch

        self.model.eval()

        result = self.forward(datas)

        dir_loss = F.mse_loss(result["real_dirs"], result["sim_dirs"])

        vae_loss = self.model.vae_real.loss_function(
            result["real_recons"],
            datas["mask"].flatten(0, 1),
            result["real_mu"],
            result["real_log_var"],
            M_N=1 / (84 * 4 * 84),
        )
        loss = dir_loss  # + vae_loss["loss"]

        self.log_dict(
            {
                "val_vae_reconstructiont": vae_loss["reconstruction_loss"],
                "val_vae_loss": vae_loss["loss"],
                "val_vae_kld": vae_loss["kld"],
                "val_dir_loss": dir_loss,
                "val_loss": loss,
            },
            sync_dist=True,
        )

    def on_validation_end(self):
        self.model.eval()

        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        device = next(self.model.parameters()).device
        test_input_dev_model = {}
        for k, v in test_input.items():
            test_input_dev_model[k] = v.to(device) if torch.is_tensor(v) else v
        result = self.forward(test_input_dev_model)
        recons = torch.nn.functional.softmax(result["real_recons"], dim=1)
        recons_class = torch.argmax(recons, dim=1)
        recons_class = recons_class.view(-1, 7, 84, 336)

        imgs_true = []
        imgs_pred = []
        for j in range(test_input["mask"].shape[1]):
            true_seg = torch.stack(
                [test_input["mask"][0, j] == cls for cls in range(4)], dim=0
            )
            true_overlay = torchvision.utils.draw_segmentation_masks(
                test_input["sim_raw"][0, j].permute(2, 0, 1).type(torch.uint8),
                true_seg,
                alpha=0.3,
            )
            imgs_true.append(true_overlay)

            recons_seg = torch.stack(
                [recons_class[0, j] == cls for cls in range(4)], dim=0
            )
            recons_overlay = torchvision.utils.draw_segmentation_masks(
                test_input["sim_raw"][0, j].permute(2, 0, 1).type(torch.uint8),
                recons_seg,
                alpha=0.3,
            )
            imgs_pred.append(recons_overlay)

        real_raw = test_input_dev_model["real_raw"].cpu()
        all_imgs = torch.stack(
            [img.permute(2, 0, 1) for img in real_raw[0]] + imgs_true + imgs_pred,
            dim=0,
        )
        vis_pred = torchvision.utils.make_grid(
            all_imgs,
            nrow=test_input["real"].shape[1],
            value_range=(0, 255),
            pad_value=255,
        )

        if self.logger is not None:
            self.logger.log_image(
                key="pred_sample",
                images=[vis_pred],
                caption=["Pred"],
            )

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        optims.append(optimizer)

        if self.scheduler_gamma is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(
                optims[0], gamma=self.scheduler_gamma
            )
            scheds.append(scheduler)

        return optims, scheds


def main_cli():
    MyLightningCLI(Sim2RealExperiment, Sim2RealDataModule, save_config_callback=None)


if __name__ == "__main__":
    main_cli()
