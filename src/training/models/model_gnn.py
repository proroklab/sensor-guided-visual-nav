import torch
from torch import nn
import torch_geometric
from typing import List
from torch import Tensor
from .beta_vae import BetaVAE


class ModelGNNAStarVAE(nn.Module):
    def __init__(
        self,
        comm_range: float,
        gnn_in_channels: int,
        gnn_out_channels: int,
        enc_out_layer: int,
        enc_trainable_layers: int,
        beta: float,
        enc_class_weights: List[float],
        **kwargs
    ) -> None:
        super(ModelGNNAStarVAE, self).__init__()
        self.comm_range = comm_range
        self.vae_sim = BetaVAE(
            gnn_in_channels,
            beta=beta,
            enc_out_layer=enc_out_layer,
            loss_type="H",
            recons_loss_type="cross_entropy",
            enc_class_weights=enc_class_weights,
        )
        self.vae_real = BetaVAE(
            gnn_in_channels,
            beta=beta,
            enc_out_layer=enc_out_layer,
            loss_type="H",
            recons_loss_type="cross_entropy",
            enc_class_weights=enc_class_weights,
        )

        for param in self.vae_sim.encoder.parameters():
            param.requires_grad = False
        for param in self.vae_real.parameters():
            param.requires_grad = False
        if enc_trainable_layers > 0:
            for param in self.vae_sim.encoder.features[
                enc_out_layer - enc_trainable_layers + 1 : enc_out_layer + 1
            ].parameters():
                param.requires_grad = True

        self.gnn_pre = torch.nn.Sequential(
            torch.nn.Linear(gnn_in_channels, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, gnn_in_channels),
        )

        self.gnn = torch_geometric.nn.conv.GraphConv(
            in_channels=gnn_in_channels,
            out_channels=gnn_out_channels,
            aggr="add",
        )

        self.post_proc = torch.nn.Sequential(
            torch.nn.Linear(gnn_out_channels, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 16),
        )

    def forward(self, input: Tensor, vae_type: str = "sim", **kwargs) -> List[Tensor]:
        img_flat = input["img_norm"].view(-1, *input["img_norm"].shape[2:])
        img_in = img_flat.permute(0, 3, 1, 2)
        vae_model = {"sim": self.vae_sim, "real": self.vae_real}[vae_type]
        self.recons, self.input, self.mu, self.log_var = vae_model.forward(img_in)

        # sample = vae_model.reparameterize(self.mu, self.log_var)
        # gnn_in = self.gnn_pre(sample)

        gnn_in = self.gnn_pre(self.mu)

        batch_size = input["img_norm"].shape[0]
        self.n_agents = input["img_norm"].shape[1]

        graphs = torch_geometric.data.Batch()
        graphs.batch = torch.repeat_interleave(
            torch.arange(batch_size), self.n_agents, dim=0
        ).to(input["img_norm"].device)
        graphs.pos = input["sensors_pos"][:, :, :2].reshape(
            batch_size * self.n_agents, 2
        )
        graphs.x = gnn_in
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
        )

        gnn_out = self.gnn.forward(graphs.x, graphs.edge_index)
        output = self.post_proc(gnn_out)

        return output.view(*input["img_norm"].shape[:2], 16)
