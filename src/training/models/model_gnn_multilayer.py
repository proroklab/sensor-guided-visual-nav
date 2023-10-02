import torch
from torch import nn
import torch_geometric
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn import Sequential, JumpingKnowledge
from typing import List
from torch import Tensor
from .beta_vae import BetaVAE


class ModelGNNMultiLayer(nn.Module):
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
        super(ModelGNNMultiLayer, self).__init__()
        self.comm_range = comm_range
        self.gnn_in_channels = gnn_in_channels
        self.gnn_out_channels = gnn_out_channels
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

        self.gnn = Sequential(
            "x, edge_index",
            [
                (nn.Dropout(p=0.2), "x -> x"),
                (
                    GraphConv(self.gnn_in_channels, self.gnn_out_channels),
                    "x, edge_index -> x1",
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.gnn_out_channels, self.gnn_out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.gnn_out_channels, self.gnn_out_channels),
                nn.ReLU(inplace=True),
                (
                    GraphConv(self.gnn_out_channels, self.gnn_out_channels),
                    "x1, edge_index -> x2",
                ),
                nn.ReLU(inplace=True),
                nn.Linear(self.gnn_out_channels, self.gnn_out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.gnn_out_channels, self.gnn_out_channels),
                nn.ReLU(inplace=True),
                (lambda x1, x2: [x1, x2], "x1, x2 -> xs"),
                (JumpingKnowledge("cat", self.gnn_out_channels), "xs -> x"),
                nn.Linear(2 * self.gnn_out_channels, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 16),
            ],
        )

    def forward(self, input: Tensor, vae_type: str = "sim", **kwargs) -> List[Tensor]:
        img_flat = input["img_norm"].view(-1, *input["img_norm"].shape[2:])
        img_in = img_flat.permute(0, 3, 1, 2)
        vae_model = {"sim": self.vae_sim, "real": self.vae_real}[vae_type]
        self.recons, self.input, self.mu, self.log_var = vae_model.forward(img_in)

        batch_size = input["img_norm"].shape[0]
        self.n_agents = input["img_norm"].shape[1]

        graphs = torch_geometric.data.Batch()
        graphs.batch = torch.repeat_interleave(
            torch.arange(batch_size), self.n_agents, dim=0
        ).to(input["img_norm"].device)
        graphs.pos = input["sensors_pos"][:, :, :2].reshape(
            batch_size * self.n_agents, 2
        )
        graphs.x = self.mu
        graphs.edge_index = torch_geometric.nn.pool.radius_graph(
            graphs.pos, batch=graphs.batch, r=self.comm_range, loop=False
        )
        output = self.gnn(graphs.x, graphs.edge_index)
        return output.view(*input["img_norm"].shape[:2], 16)
