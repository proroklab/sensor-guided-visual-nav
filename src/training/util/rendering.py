import torch
import torchvision
import json
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from ...dataset_util.src.env.env_v2 import Env


def render_astar_grid(test_input, test_label, result, data_path):
    batch_rendering = []
    for data_id, labels_vec, results_vec in zip(test_input["id"], test_label, result):
        with open(data_path / "meta" / (data_id + ".json"), "r") as file:
            meta = json.load(file)
        env = Env(meta["cfg"])
        env.restore_state(meta)

        fig = plt.figure(
            figsize=(env.cfg["env_size_boxes"][1], env.cfg["env_size_boxes"][0]),
            constrained_layout=True,
        )
        ax = fig.add_subplot(111)

        env.render(ax, render_only_direct_path=False)

        sensor_grid_coords = []
        for box, label_vec, result_vec in zip(
            meta["robots"] + meta["sensors"], labels_vec, results_vec
        ):
            circle_r_m = 0.25
            ax.add_patch(
                plt.Circle(
                    box["t"][:2],
                    circle_r_m,
                    color="black",
                    fill=False,
                )
            )

            angles = torch.arange(0, 2 * torch.pi, torch.pi / 8) - box["r"][3]
            for values, color, offset in [
                [result_vec, "orange", [0, 0]],
                [label_vec, "lightblue", [0.02, 0.02]],
            ]:
                for value, angle in zip(values, angles):
                    vec = torch.Tensor([torch.cos(angle), torch.sin(angle)])
                    start_point = (
                        torch.Tensor([box["t"][0], box["t"][1]])
                        + vec * circle_r_m
                        + torch.Tensor(offset)
                    )
                    end_point = (
                        torch.Tensor([box["t"][0], box["t"][1]])
                        + vec * (circle_r_m + value)
                        + torch.Tensor(offset)
                    )
                    ax.plot(
                        [start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        c=color,
                        linewidth=3,
                    )

        ax.axis("off")

        fig.canvas.draw()

        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        batch_rendering.append(
            torch.from_numpy(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        )

    return torchvision.utils.make_grid(
        torch.stack(batch_rendering, dim=0).permute(0, 3, 1, 2) / 255.0,
        nrow=4,
        value_range=(0.0, 1.0),
        pad_value=1.0,
    )
