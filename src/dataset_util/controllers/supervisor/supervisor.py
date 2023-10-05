from controller import Supervisor
import cv2
from pathlib import Path
import json
import argparse
from ...src.env.env_webot_utils import (
    query_picture_from_sensor,
    reset_boxes,
    reset_sensors_target,
    initialize_scene,
)

TIME_STEP = 32

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Dataset")
    parser.add_argument("dataset_path")
    parser.add_argument("--start_file_index", nargs="?", type=int, default=0)
    parser.add_argument("--end_file_index", nargs="?", type=int, default=-1)
    parser.add_argument("--sensor_port_start", nargs="?", type=int, default=8000)
    args = parser.parse_args()

    robot = Supervisor()

    timestep = 0
    dataset_path = Path(args.dataset_path)
    files = sorted((dataset_path / "meta").glob("*.json"))
    current_file_index = args.start_file_index
    while robot.step(TIME_STEP) != -1:
        print(timestep)
        if timestep % 3 == 0:
            data_id = files[current_file_index].stem
            print(data_id)
            if current_file_index == len(files):
                break

            with open(files[current_file_index], "r") as infile:
                meta = json.load(infile)
            n_sensors = len(meta["sensors"]) + 1

            if timestep == 0:
                initialize_scene(
                    robot,
                    len(meta["sensors"]),
                    port_start=args.sensor_port_start,
                )
                for i in range(n_sensors):
                    (dataset_path / f"sensor_spherical_{i}").mkdir(
                        parents=True, exist_ok=True
                    )

            reset_boxes(robot, meta["border_obstacles"] + meta["inner_obstacles"])
            reset_sensors_target(
                robot, meta["sensors"], meta["targets"][0], meta["robots"][0]
            )

        if timestep % 3 == 2:
            for i in range(n_sensors):
                print("Q", i)
                frame = query_picture_from_sensor(args.sensor_port_start + i)
                if frame is not None:
                    cv2.imwrite(
                        str(dataset_path / f"sensor_spherical_{i}" / f"{data_id}.png"),
                        frame,
                    )
            current_file_index += 1
            if args.end_file_index > 0 and current_file_index >= args.end_file_index:
                break

        timestep += 1
