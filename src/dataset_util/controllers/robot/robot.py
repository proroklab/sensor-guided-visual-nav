from controller import Supervisor
import numpy as np
import cv2
import yaml
import json
import argparse
from pathlib import Path
from pytorch_lightning.utilities.seed import seed_everything
from ...src.util.util import get_model_class
import torch
import torchvision.transforms as T
from collections import OrderedDict
from ...src.env.env_webot_utils import (
    query_picture_from_sensor,
    reset_boxes,
    reset_sensors_target,
    initialize_scene,
)


TIME_STEP = 32
PLACE_ROBOT = True
base_seed = 0
initial_index = 0

##################################

parser = argparse.ArgumentParser(description="Generic runner for VAE models")
parser.add_argument(
    "checkpoint_file",
)
parser.add_argument(
    "dataset_path",
)
parser.add_argument("data_id", type=int)
parser.add_argument(
    "evaluation_path",
)
parser.add_argument("--sensor_port_start", nargs="?", type=int, default=8000)

args = parser.parse_args()


class BasePolicy:
    def __init__(self):
        pass

    def step(obs):
        return None


class ImitationLearningPolicy(BasePolicy):
    def __init__(self, checkpoint_file):
        super().__init__()

        with open(checkpoint_file.parent / "config.yaml", "r") as file:
            self.config = yaml.safe_load(file)
            # self.config["model_params"]["comm_range"] = 2.0
            seed_everything(self.config["exp_params"]["manual_seed"], True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model_class(self.config["model"])(**self.config["model_params"])
        self.model = self.load_model(checkpoint_file, model).to(self.device)
        self.model.eval()

    @staticmethod
    def load_model(checkpoint_path, model):
        model_state = torch.load(checkpoint_path, map_location=torch.device("cpu"))[
            "state_dict"
        ]

        # A basic remapping is required
        mapping = {k: v for k, v in zip(model_state.keys(), model.state_dict().keys())}
        mapped_model_state = OrderedDict(
            [(mapping[k], v) for k, v in model_state.items()]
        )
        model.load_state_dict(mapped_model_state, strict=False)
        return model

    def step(self, obs):
        with torch.no_grad():
            frames_raw = obs["frames"]
            t_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            frames = []
            for frame in frames_raw:
                frame_tensor = torch.Tensor(frame)
                img_norm = t_norm(frame_tensor.permute(2, 0, 1) / 255.0).permute(
                    1, 2, 0
                )
                frames.append(img_norm)

            datapoint = {
                "img_norm": torch.stack(frames, dim=0).unsqueeze(0).to(self.device),
                "sensors_pos": torch.Tensor(obs["pos"]).unsqueeze(0).to(self.device),
            }
            results = self.model.forward(datapoint).squeeze(0).cpu()

            delta_ps = torch.Tensor(
                [
                    [torch.cos(a), torch.sin(a)]
                    for a in torch.arange(0, 2 * torch.pi, torch.pi / 8)
                ]
            )

            sm_robot_result = torch.nn.Softmax(dim=0)(-results[0] * 40)
            direction = torch.distributions.categorical.Categorical(
                sm_robot_result
            ).sample()
            # direction = torch.argmin(results, dim=1)
            normed_movement = delta_ps[direction]

            # raw_movement = (delta_ps * -results.unsqueeze(2)).sum(dim=1)
            # normed_movement = raw_movement / torch.linalg.norm(raw_movement, dim=1).unsqueeze(1)

            return normed_movement


def get_sensor_frames(n_sensors, port_start=8000):
    frames = []
    for i in range(n_sensors):
        img_circ = query_picture_from_sensor(port_start + i)
        if img_circ is None:
            return None
        img_polar = cv2.warpPolar(
            img_circ,
            (336, 336),
            np.array(img_circ.shape[:2]) / 2,
            336,
            cv2.WARP_FILL_OUTLIERS,
        )[:, 0:168]
        img_polar_rot = cv2.rotate(img_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_polar_rot = cv2.resize(img_polar_rot, (336, 84))

        img_polar_rot_bgr = cv2.cvtColor(img_polar_rot, cv2.COLOR_BGR2RGB)
        frames.append(img_polar_rot_bgr)
    return frames


robot = Supervisor()
rng = np.random.default_rng(1)
policy = ImitationLearningPolicy(Path(args.checkpoint_file))
eval_path = Path(args.evaluation_path)
eval_path.mkdir(parents=True, exist_ok=True)

episode_timestep = 0
dataset_path = Path(args.dataset_path)
episode_files = sorted((dataset_path / "meta").glob("*.json"))

with open(episode_files[args.data_id], "r") as infile:
    current_meta = json.load(infile)
data_id = episode_files[args.data_id].stem

initialize_scene(robot, len(current_meta["sensors"]), port_start=args.sensor_port_start)

reset_boxes(robot, current_meta["border_obstacles"] + current_meta["inner_obstacles"])
reset_sensors_target(
    robot,
    current_meta["sensors"],
    current_meta["targets"][0],
    current_meta["robots"][0],
)

robot_path = []
while robot.step(TIME_STEP) != -1:
    n_sensors = len(current_meta["sensors"])
    frames = get_sensor_frames(n_sensors + 1, port_start=args.sensor_port_start)
    if frames:
        sensors_pos = []
        sensors_pos.append(robot.getFromDef("robot").getPosition()[:2])
        for i in range(1, n_sensors + 1):
            sensor_node = robot.getFromDef(f"sensor_{i}")
            sensors_pos.append(sensor_node.getPosition()[:2])

        movement = (
            policy.step({"meta": current_meta, "frames": frames, "pos": sensors_pos})
            * 0.1
        )
        new_pose = torch.Tensor(sensors_pos[0]) + movement
        robot_path.append(new_pose.tolist())

        robot_node = robot.getFromDef("robot")
        target_node = robot.getFromDef("target")
        robot_node.getField("translation").setSFVec3f(new_pose.tolist() + [0.0])
        dst_goal = np.linalg.norm(
            np.array(robot_node.getPosition()) - np.array(target_node.getPosition())
        )
        if episode_timestep > 200 or dst_goal < 0.5:
            episode_timestep = 0
            with open(eval_path / f"{data_id}.json", "w") as outfile:
                json.dump(robot_path, outfile)
            break
        else:
            episode_timestep += 1

        print(episode_timestep)
