import torch
import yaml
import json
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable, Dict
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
import random
import pandas as pd
import numpy as np
import yaml
from PIL import Image
import torchvision.transforms as T


class Sim2RealDataset(Dataset):
    def __init__(
        self,
        data_path_str: str,
        split_start: float,
        split_end: float,
        dataset: str,
        **kwargs,
    ):
        self.data_path = Path(data_path_str)

        # with open(self.data_path / "meta" / "shifts.json", "r") as infile:
        #    shifts = json.load(infile)

        data_files = []
        if dataset == "train":
            data_ep_ranges = ["episode_1[4-9]", "episode_2[0-9]", "episode_30"]
        elif dataset == "eval":
            data_ep_ranges = ["episode_1[0-3]"]
        elif dataset == "test":
            data_ep_ranges = ["episode_15"]
        else:
            assert False

        # poses = {"robot": {}, "target": {}}
        # for pos_csv_file in self.data_path.glob(
        #    f"{data_ep_range}/sensor_0/current_state/pos.csv"
        # ):
        #    print(pos_csv_file)
        #    sensor = pos_csv_file.parent.parent.stem
        #    poses[sensor] = pd.read_csv(pos_csv_file)[["x", "y"]].to_numpy()

        for data_ep_range in data_ep_ranges:
            for sim_file in sorted(
                self.data_path.glob(f"{data_ep_range}/sensor_0/image_sim_sync/*.png")
            ):
                episode_dir = sim_file.parent.parent.parent
                sample_files = []
                for sens_folder in sorted(
                    sim_file.parent.parent.parent.glob("sensor_*")
                ):
                    real_file = sens_folder / "image_real_sync" / sim_file.name
                    sim_file = sens_folder / "image_sim_sync" / sim_file.name
                    mask_file = sens_folder / "image_sim_sync_mask" / sim_file.name
                    if (
                        not real_file.is_file()
                        or not sim_file.is_file()
                        or not mask_file.is_file()
                    ):
                        break
                    sensor = sim_file.parent.parent.stem
                    if sensor == "sensor_0":
                        sim_shift = 168
                    else:
                        sim_shift = 0
                    # idx = int(sim_file.stem)
                    sample_files.append(
                        {
                            "sim": sim_file,
                            "real": real_file,
                            "mask": mask_file,
                            "real_shift": sim_shift,
                            "sim_shift": sim_shift,  # int(shifts[episode_dir.stem][sens_folder.stem]),
                        }
                    )
                if len(sample_files) == 7:
                    data_files.append(sample_files)

        self.data_files = data_files
        print(len(data_files), len(self.data_files))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_file = self.data_files[idx]

        jitter = T.ColorJitter(
            brightness=(0.5, 1.5), hue=0.05, contrast=0.1, saturation=0.05
        )
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        imgs_real_raw = []
        imgs_real = []
        imgs_sim_raw = []
        imgs_sim = []
        imgs_mask = []
        for sensor_meta in data_file:
            img_sim = Tensor(
                np.array(Image.open(sensor_meta["sim"]).convert("RGB"))
            ).permute(2, 0, 1)
            img_real = Tensor(
                np.array(Image.open(sensor_meta["real"]).convert("RGB"))
            ).permute(2, 0, 1)
            imgs_mask.append(
                Tensor(np.array(Image.open(sensor_meta["mask"]).convert("L"))).long()
            )

            img_sim_roll = torch.roll(img_sim, sensor_meta["sim_shift"], dims=2)
            imgs_sim_raw.append(img_sim_roll)

            img_real_roll = torch.roll(img_real, sensor_meta["real_shift"], dims=2)
            imgs_real_raw.append(img_real_roll)

            img_real_aug = jitter(img_real_roll / 255.0)

            imgs_real.append(norm(img_real_aug))
            imgs_sim.append(norm(img_sim_roll / 255.0))

        return {
            "sensors_pos": torch.zeros(7, 2),
            "sim_raw": torch.stack(imgs_sim_raw, dim=0).permute(0, 2, 3, 1),
            "real_raw": torch.stack(imgs_real_raw, dim=0).permute(0, 2, 3, 1),
            "sim": torch.stack(imgs_sim, dim=0).permute(0, 2, 3, 1),
            "real": torch.stack(imgs_real, dim=0).permute(0, 2, 3, 1),
            "mask": torch.stack(imgs_mask, dim=0),
        }, []


class Sim2RealDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        num_workers: int = 0,
        split_fractions: Dict = {},
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_fractions = split_fractions
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = Sim2RealDataset(
            self.data_dir,
            self.split_fractions["train"][0],
            self.split_fractions["train"][1],
            dataset="train",
        )

        self.val_dataset = Sim2RealDataset(
            self.data_dir,
            self.split_fractions["eval"][0],
            self.split_fractions["eval"][1],
            dataset="eval",
        )

        self.test_dataset = Sim2RealDataset(
            self.data_dir,
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
