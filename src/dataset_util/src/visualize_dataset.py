from pathlib import Path
from PIL import Image
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from matplotlib.patches import ConnectionPatch
import matplotlib.patches as patches
import matplotlib as mpl
from ..src.env.env_v2 import Env

parser = argparse.ArgumentParser(description="Convert Dataset")
parser.add_argument("dataset_path")
parser.add_argument("dataset_item", type=int)
parser.add_argument("filename", nargs="?", default=None)
args = parser.parse_args()

data_path = Path(args.dataset_path)


def annotate_axes(ax, text, fontsize=18):
    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="darkgrey",
    )


def grid_rect_patch(ax, pos, size, angle, color):
    origin = np.array(pos) - np.array(size) / 2
    rect = patches.Rectangle(
        (origin[1], origin[0]),
        size[1],
        size[0],
        linewidth=1,
        edgecolor=color,
        facecolor="none",
    )
    transform = (
        mpl.transforms.Affine2D().rotate_around(pos[1], pos[0], -angle) + ax.transData
    )
    rect.set_transform(transform)
    ax.add_patch(rect)


def visualize(data_id):
    print(data_id)
    datapoint = {}
    with open(data_path / "meta" / (data_id + ".json"), "r") as file:
        meta = json.load(file)

    env = Env(meta["cfg"])
    env.restore_state(meta)

    sensor_img_rotations = {"raw": 180, "spherical": 180, "polar": 0}
    sensor_imgs = {"raw": {}, "spherical": {}, "polar": {}}
    for img_type_key, imgs in sensor_imgs.items():
        for sensor_path in sorted(data_path.glob(f"sensor_{img_type_key}_*")):
            img = Image.open(sensor_path / (data_id + ".png")).convert("RGB")
            imgs[sensor_path.stem] = np.array(
                img.rotate(sensor_img_rotations[img_type_key])
            )

    outer = [["upper"] * max(1, len(sensor_imgs["raw"]))]
    for imgs in sensor_imgs.values():
        if len(imgs) > 0:
            outer += [list(imgs.keys())]

    fig, axd = plt.subplot_mosaic(outer, constrained_layout=True, figsize=(12, 6))

    for img_type_key, imgs in sensor_imgs.items():
        for (sensor_id, sensor_img), sensor_meta in zip(
            imgs.items(), meta["sensors"] + meta["robots"]
        ):
            axd[sensor_id].imshow(sensor_img)
            if img_type_key == "raw":
                axd[sensor_id].add_artist(
                    ConnectionPatch(
                        xyA=(sensor_meta["t"][0], sensor_meta["t"][1]),
                        xyB=(0.5, 0),
                        coordsA="data",
                        coordsB="data",
                        axesA=axd["upper"],
                        axesB=axd[sensor_id],
                        color="blue",
                    )
                )

    env.render(axd["upper"], render_only_direct_path=False)
    if args.filename:
        fig.savefig(args.filename, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


visualize(f"{args.dataset_item:06d}")
