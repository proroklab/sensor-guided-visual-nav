import torch
import yaml
import json
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable, Dict
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule, LightningDataModule
from torch.utils.data import DataLoader

import numpy as np
import yaml
from PIL import Image


import torchvision.transforms as T


def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma_max = 0.03
    noise_amplitude = torch.rand(1).item()

    out = img + sigma_max * noise_amplitude * torch.randn_like(img)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


class SimDataset(Dataset):
    def __init__(
        self,
        data_path_str: str,
        data_type: str,
        split_start: float,
        split_end: float,
        load_images: bool = True,
        omit_invalid_paths: bool = True,
        **kwargs,
    ):
        assert data_type in ["raw", "spherical", "polar"]
        self.data_type = data_type
        self.load_images = load_images
        self.omit_invalid_paths = omit_invalid_paths
        self.data_path = Path(data_path_str)
        labels_files = list(sorted((self.data_path / "meta").glob("*.json")))

        data_ids = []
        # data_ids = [f.stem for f in labels_files]
        for meta_file in labels_files:
            with open(meta_file, "r") as file:
                meta = json.load(file)
            sensors_pos = [s["t"] for s in meta["robots"] + meta["sensors"]]
            any_zero = False
            for i in range(len(sensors_pos)):
                path_info = (meta["robot_paths"] + meta["sensor_paths"])[i][0]
                path = path_info["path"]
                if self.omit_invalid_paths and len(path) == 0:
                    any_zero = True
                    break
            if not any_zero:
                data_ids.append(meta_file.stem)
            # if len(data_ids) > 200:
            #    break

        data_start = int(split_start * len(data_ids))
        data_end = int(split_end * len(data_ids))
        self.data_ids = data_ids[data_start:data_end]
        print(len(data_ids), len(self.data_ids))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        meta_file = self.data_path / "meta" / (data_id + ".json")
        data_id = meta_file.stem
        datapoint = {"id": data_id}
        with open(meta_file, "r") as file:
            meta = json.load(file)

        sensors_pos = [s["t"] for s in meta["robots"] + meta["sensors"]]
        datapoint["sensors_pos"] = Tensor(sensors_pos)
        datapoint["target_pos"] = Tensor(meta["targets"][0]["t"])

        label_list = []
        for i in range(len(sensors_pos)):
            path_costs = []
            for path_info in (meta["robot_paths"] + meta["sensor_paths"])[i]:
                path = path_info["path"]
                if len(path) > 1:
                    cost = np.linalg.norm(
                        (np.roll(path, -1, axis=0) - path)[:-1], axis=1
                    ).sum()
                else:
                    if self.omit_invalid_paths:
                        assert len(path_costs) > 0, f"{data_id}"
                    if len(path_costs) == 0:
                        cost = 0.25
                    else:
                        cost = path_costs[0] + 0.25
                path_costs.append(cost)
            cost_advantages = torch.Tensor(path_costs[1:]) - path_costs[0]
            rot = (meta["robots"] + meta["sensors"])[i]["r"][3]
            idx_roll = int((rot / 3.1415926535) * len(cost_advantages) / 2)
            label_list.append(cost_advantages.roll(idx_roll))

        label = torch.stack(label_list, dim=0)

        imgs = []
        imgs_norm = []
        segs = []
        t_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        jitter = T.ColorJitter(
            brightness=(0.5, 1.5), hue=0.05, contrast=0.1, saturation=0.05
        )
        if self.load_images:
            for i in range(len(sensors_pos)):
                img_camera = Image.open(
                    self.data_path / f"sensor_{self.data_type}_{i}" / (data_id + ".png")
                )
                img_conv = Tensor(np.array(img_camera.convert("RGB"))) / 255.0
                img_conv = jitter(img_conv.permute(2, 0, 1)).permute(1, 2, 0)
                img_conv = gauss_noise_tensor(img_conv)
                imgs.append(img_conv)
                img_norm = t_norm(img_conv.permute(2, 0, 1)).permute(1, 2, 0)
                imgs_norm.append(img_norm)
                img_seg = Image.open(
                    self.data_path
                    / "masks"
                    / f"sensor_{self.data_type}_{i}"
                    / (data_id + ".png")
                )
                segs.append(Tensor(np.array(img_seg.convert("L"))))
            datapoint["img_raw"] = torch.stack(imgs, dim=0)
            datapoint["img_norm"] = torch.stack(imgs_norm, dim=0)
            datapoint["img_seg"] = torch.stack(segs, dim=0).long()
        return datapoint, label


class SimDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        data_type: str,
        batch_size: int = 8,
        num_workers: int = 0,
        split_fractions: Dict = {},
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.data_type = data_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_fractions = split_fractions
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SimDataset(
            self.data_dir,
            self.data_type,
            self.split_fractions["train"][0],
            self.split_fractions["train"][1],
            dataset="train",
        )

        self.val_dataset = SimDataset(
            self.data_dir,
            self.data_type,
            self.split_fractions["eval"][0],
            self.split_fractions["eval"][1],
            dataset="eval",
        )

        self.test_dataset = SimDataset(
            self.data_dir,
            self.data_type,
            self.split_fractions["test"][0],
            self.split_fractions["test"][1],
            dataset="test",
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
