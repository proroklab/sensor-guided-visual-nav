import torch
import torchvision
import json
import torch
from torch import optim, Tensor
from torch.nn import functional as F
from lightning.pytorch import LightningModule
import matplotlib as mpl
import cv2
from typing import Optional

mpl.use("agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
from src.dataset_util.src.env.env_v2 import Env
from src.training.util.rendering import render_astar_grid
from src.training.util.cli import MyLightningCLI
from src.training.dataloaders.dataset_sim import SimDataModule


class AStarGridVAEExperiment(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        scheduler_gamma: Optional[float] = 0.98,
    ) -> None:
        super(AStarGridVAEExperiment, self).__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma

    def forward(self, input) -> Tensor:
        return self.model(input)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        datas, labels = batch
        self.model.train()
        results = self.forward(datas)
        # https://github.com/AntixK/PyTorch-VAE/issues/56
        vae_loss = self.model.vae_sim.loss_function(
            self.model.recons,
            datas["img_seg"].flatten(0, 1),
            self.model.mu,
            self.model.log_var,
            M_N=1 / (84 * 84 * 4),
        )
        nav_loss = F.mse_loss(results, labels)

        nav_l1 = F.l1_loss(results, labels)
        nav_robot_l1 = F.l1_loss(results[:, 0], labels[:, 0])

        loss = nav_loss + vae_loss["loss"]

        self.log_dict(
            {
                "loss": loss,
                "nav_loss": nav_loss,
                "nav_l1": nav_l1,
                "nav_robot_l1": nav_robot_l1,
                "vae_loss": vae_loss["loss"],
                "vae_reconstruction": vae_loss["reconstruction_loss"],
                "vae_c": vae_loss["C"],
                "vae_kld": vae_loss["kld"],
            },
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        datas, labels = batch
        self.model.eval()
        results = self.forward(datas)

        # https://github.com/AntixK/PyTorch-VAE/issues/56
        vae_loss = self.model.vae_sim.loss_function(
            self.model.recons,
            datas["img_seg"].flatten(0, 1),
            self.model.mu,
            self.model.log_var,
            M_N=1 / (84 * 84 * 4),
        )

        nav_loss = F.mse_loss(results, labels)

        nav_l1 = F.l1_loss(results, labels)
        nav_robot_l1 = F.l1_loss(results[:, 0], labels[:, 0])

        loss = nav_loss + vae_loss["loss"]

        self.log_dict(
            {
                "val_loss": loss,
                "val_nav_loss": nav_loss,
                "val_nav_l1": nav_l1,
                "val_nav_robot_l1": nav_robot_l1,
                "val_vae_loss": vae_loss["loss"],
                "val_vae_c": vae_loss["C"],
                "val_vae_reconstruction": vae_loss["reconstruction_loss"],
                "val_vae_kld": vae_loss["kld"],
            },
            sync_dist=True,
        )

    def on_validation_end(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        device = next(self.model.parameters()).device
        test_input_dev_model = {}
        for k, v in test_input.items():
            test_input_dev_model[k] = v.to(device) if torch.is_tensor(v) else v
        result = self.forward(test_input_dev_model).cpu()

        data_path = Path(self.trainer.datamodule.data_dir)
        all_recons = self.model.recons.detach().view(
            -1, self.model.n_agents, *self.model.recons.shape[1:]
        )
        all_recons_s = torch.nn.functional.softmax(all_recons, dim=2)
        all_recons_class = torch.argmax(all_recons_s, dim=2)

        imgs_raw = []
        imgs_true_overlaid = []
        imgs_pred_overlaid = []
        for i in range(self.model.n_agents):
            true_seg = torch.stack(
                [test_input["img_seg"][0, i] == cls for cls in range(4)], dim=0
            )
            img_raw = (test_input["img_raw"][0, i].permute(2, 0, 1) * 255).type(
                torch.uint8
            )
            imgs_raw.append(img_raw)
            true_overlay = torchvision.utils.draw_segmentation_masks(
                img_raw,
                true_seg,
                alpha=0.5,
            )
            imgs_true_overlaid.append(true_overlay)

            recons_seg = torch.stack(
                [all_recons_class[0, i] == cls for cls in range(4)], dim=0
            )
            recons_overlay = torchvision.utils.draw_segmentation_masks(
                img_raw,
                recons_seg,
                alpha=0.5,
            )
            imgs_pred_overlaid.append(recons_overlay)

        vae_recons = (
            torchvision.utils.make_grid(
                imgs_raw + imgs_true_overlaid + imgs_pred_overlaid,
                nrow=self.model.n_agents,
                value_range=(0, 255),
                pad_value=255,
            )
            / 255.0
        )

        vis_pred = render_astar_grid(test_input, test_label, result, data_path)

        if self.logger is not None:
            self.logger.log_image(
                key="pred_sample",
                images=[vis_pred],
                caption=["Pred"],
            )
            self.logger.log_image(
                key="vae_recons",
                images=[vae_recons],
                caption=["Reconstruction"],
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
    MyLightningCLI(AStarGridVAEExperiment, SimDataModule, save_config_callback=None)


if __name__ == "__main__":
    main_cli()
