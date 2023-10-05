import scipy.ndimage
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
import multiprocessing.pool as mpp
from multiprocessing import Pool
import pyvisgraph as vg
import shapely

from .env.env_v2 import Env


def convert_file(meta_file, n_paths, path_start_dist, min_dist_obs):
    data_id = meta_file.stem

    with open(meta_file, "r") as file:
        meta = json.load(file)

    angles = np.linspace(0, 2 * np.pi, n_paths + 1)[:-1]
    delta_ps = (
        np.array([[0.0, 0.0]] + [[np.cos(a), np.sin(a)] for a in angles])
        * path_start_dist
    )

    if min_dist_obs is None:
        min_dist_obs = meta["cfg"]["obstacle_clearance_sensor_robot"]
    else:
        meta["cfg"]["obstacle_clearance_sensor_robot"] = min_dist_obs

    env = Env(meta["cfg"])
    env.restore_state(meta)

    sensor_occupied_area = shapely.ops.unary_union(
        [box["shape"] for box in env.border_boxes + env.obstacle_boxes]
    )

    env.sensor_paths = env.compute_paths(
        env.target_boxes[0],
        env.sensor_boxes,
        sensor_occupied_area,
        env.cfg["obstacle_clearance_sensor_robot"],
        delta_ps,
    )

    robot_occupied_area = shapely.ops.unary_union(
        [
            box["shape"]
            for box in env.border_boxes + env.obstacle_boxes + env.sensor_boxes
        ]
    )

    env.robot_paths = env.compute_paths(
        env.target_boxes[0],
        env.robot_boxes,
        robot_occupied_area,
        env.cfg["obstacle_clearance_sensor_robot"],
        delta_ps,
    )

    meta = env.save_state()
    with open(meta_file.parent.parent / "meta" / (data_id + ".json"), "w") as outfile:
        json.dump(meta, outfile)


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Dataset")
    parser.add_argument("dataset_path")
    parser.add_argument(
        "--min_dist_obs", type=float, default=None, help="Minimum distance to wall"
    )
    parser.add_argument(
        "--path_start_dist",
        type=float,
        default=0.2,
        help="Distance for path to start around robot",
    )
    parser.add_argument(
        "--n_workers", type=int, default=16, help="Number of parallel workers"
    )
    parser.add_argument(
        "--n_paths", type=int, default=8, help="Number of paths around robot to create"
    )
    parser.add_argument("--start_index", type=int, default=0, help="Start index")
    parser.add_argument("--end_index", type=int, default=-1, help="End index")
    parser.add_argument(
        "--n_workers",
        type=int,
        default=1,
        help="Number of parallel workers to build visgraph",
    )
    args = parser.parse_args()

    all_files = sorted((Path(args.dataset_path) / "meta").glob("*.json"))
    if args.end_index > 0:
        files = all_files[args.start_index : args.end_index]
    else:
        files = all_files
    with Pool(args.n_workers) as pool:
        iterable = [
            (file, args.n_paths, args.path_start_dist, args.min_dist_obs)
            for file in files
        ]
        for _ in tqdm(pool.istarmap(convert_file, iterable), total=len(iterable)):
            pass
