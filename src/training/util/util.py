import torch
from ..models.model_gnn import ModelGNNAStarVAE
from ..models.model_gnn_multilayer import ModelGNNMultiLayer


def wrap_angle(x, min, max):
    def wrap_max(x, max):
        return torch.fmod(max + torch.fmod(x, max), max)

    return min + wrap_max(x - min, max - min)


def get_model_class(identifier):
    return {
        "gnn_astar_vae": ModelGNNAStarVAE,
        "gnn_mult": ModelGNNMultiLayer,
    }[identifier]
