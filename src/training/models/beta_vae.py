"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/
			    Copyright Anand Krishnamoorthy Subramanian 2020
			               anandkrish894@gmail.com

Modified from https://github.com/AntixK/PyTorch-VAE/blob/db41da7f1d68308832ea52098cb36a149b919c45/models/beta_vae.py
"""


import torch
from torch import nn
from typing import List
from torch import Tensor
import torchvision
from torch.nn import functional as F
from .dice_loss import DiceLoss


class BetaVAE(nn.Module):
    num_iter = 0  # Global static variable to keep track of iterations

    def __init__(
        self,
        latent_dim,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "H",
        recons_loss_type: str = "cross_entropy",
        enc_out_layer: int = -1,
        enc_class_weights=[1, 1, 1],
    ) -> None:
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.recons_loss_type = recons_loss_type
        self.enc_out_layer = enc_out_layer
        self.enc_class_weights = enc_class_weights
        out_channels = len(enc_class_weights)

        self.dice_loss = DiceLoss("multiclass", [0, 1, 2, 3])

        self.encoder = torchvision.models.mobilenet_v2(pretrained=True)

        self.encoder_output_layer = enc_out_layer
        enc_out_channels = self.encoder.features[self.enc_out_layer].out_channels
        self.enc_post = torch.nn.Sequential(
            torch.nn.Linear(enc_out_channels, self.latent_dim),
            torch.nn.LeakyReLU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                int(self.latent_dim / 4),
                64,
                5,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=0, output_padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=0, output_padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 16, 5, stride=3, padding=0, output_padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2, padding=0, output_padding=0),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=0, output_padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, out_channels, 3, stride=1, padding=1),
        )

        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        if self.enc_out_layer == -1:
            x = self.encoder.features(input)
        else:
            x = self.encoder.features[: self.enc_out_layer + 1](input)

        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        enc_out = torch.nn.ReLU()(x)
        result = self.enc_post(enc_out)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = z.view(-1, int(self.latent_dim / 4), 2, 2)
        result = self.decoder(result)
        return result[:, :, 150:234, 24:360]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, recons, input, mu, log_var, **kwargs) -> dict:
        self.num_iter += 1
        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset

        if self.recons_loss_type == "mse":
            if "loss_weight" in kwargs:
                weight = kwargs["loss_weight"]
                recons_loss = (weight * (recons - input) ** 2).mean()
            else:
                recons_loss = ((recons - input) ** 2).mean()
        elif self.recons_loss_type == "cross_entropy":
            weights = torch.Tensor(self.enc_class_weights).to(input.device)
            recons_loss = F.cross_entropy(recons, input, weight=weights)
        elif self.recons_loss_type == "dice":
            recons_loss = self.dice_loss(recons, input)
        else:
            raise ValueError("Undefined reconstruction loss type.")

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        C = torch.Tensor([0.0])
        if self.loss_type == "H":  # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == "B":  # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        else:
            raise ValueError("Undefined loss type.")

        return {
            "loss": loss,
            "reconstruction_loss": recons_loss.detach(),
            "kld": kld_loss.detach(),
            "C": C.detach(),
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
