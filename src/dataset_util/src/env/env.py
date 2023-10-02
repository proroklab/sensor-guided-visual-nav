import numpy as np
import scipy.ndimage
import cv2


class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        world_size = (
            self.cfg["box_dim_m"][0] * np.array(self.cfg["env_size_boxes"])
            + self.cfg["box_dim_m"][1] * 2
        )
        self.base_grid_shape = (world_size / self.cfg["env_resolution_m"]).astype(int)

    def get_base_grid(self):
        return np.zeros(self.base_grid_shape, dtype=int)

    def world_to_grid(self, p):
        return (
            np.array(p) / self.cfg["env_resolution_m"] + self.base_grid_shape / 2
        ).astype(int)

    def grid_to_world(self, p):
        return (np.array(p) - self.base_grid_shape / 2) * self.cfg["env_resolution_m"]

    def create_box(self, pos, angle, box_dim, grid=None):
        points = box_dim[:2] * np.array(
            [
                [0.5, 0.5],
                [0.5, -0.5],
                [-0.5, -0.5],
                [-0.5, 0.5],
            ]
        )
        rot = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )

        # Swap x and y for opencv
        rotated_points = self.world_to_grid(pos + points @ rot)[:, [1, 0]]
        if grid is None:
            grid = self.get_base_grid()
        grid_edited = grid.astype(float).copy()
        cv2.fillPoly(grid_edited, pts=[rotated_points], color=1.0)
        return {
            "t": [float(pos[0]), float(pos[1]), float(box_dim[2] / 2)],
            "r": [0.0, 0.0, 1.0, float(angle)],
            "size": [float(d) for d in box_dim],
        }, grid_edited > 0.0

    def compute_border_boxes(self, box_index=1):
        boxes = []
        env_grid = self.get_base_grid()

        offset_a = (
            (np.array(self.cfg["env_size_boxes"]) - 1) / 2 * self.cfg["box_dim_m"][0]
        )
        offset_b = (
            np.array(self.cfg["env_size_boxes"]) / 2 * self.cfg["box_dim_m"][0]
            + self.cfg["box_dim_m"][1] / 2
        )
        for side in [-1, 1]:
            for i in range(self.cfg["env_size_boxes"][0] + 2):
                box, grid = self.create_box(
                    [
                        self.cfg["box_dim_m"][0] * (i - 1) - offset_a[0],
                        side * offset_b[1],
                    ],
                    0.0,
                    self.cfg["box_dim_m"],
                )
                box["i"] = int(box_index)
                boxes.append(box)
                env_grid[grid] = box_index
                box_index += 1

            for i in range(self.cfg["env_size_boxes"][1]):
                box, grid = self.create_box(
                    [
                        side * offset_b[0],
                        self.cfg["box_dim_m"][0] * i - offset_a[1],
                    ],
                    1.57,
                    self.cfg["box_dim_m"],
                )
                box["i"] = int(box_index)
                boxes.append(box)
                env_grid[grid] = box_index
                box_index += 1

        self.border_boxes, self.border_grid = boxes, env_grid

    def expand_occupancy(self, occupancy, distance):
        dst_map = scipy.ndimage.distance_transform_edt(occupancy == 0)
        return dst_map < int(distance / self.cfg["env_resolution_m"])

    def place_randomly(
        self,
        rng,
        angle_range,
        placed_boxes,
        box_dims,
        occupied_area,
        current_grid,
        min_dist,
        box_start_index=1,
    ):
        if len(placed_boxes) == len(box_dims):
            return placed_boxes, current_grid

        for i in range(100):
            possible_positions = np.stack(np.nonzero(current_grid != 1), axis=1)
            pos_grid = possible_positions[rng.integers(possible_positions.shape[0])]
            pos_world = self.grid_to_world(pos_grid)

            rand_angle = rng.random() * angle_range
            current_box = len(placed_boxes)
            box, grid = self.create_box(
                pos_world, rand_angle, box_dims[current_box], self.get_base_grid()
            )
            dst_map = scipy.ndimage.distance_transform_edt(grid != 1)
            extended_grid = dst_map < int(min_dist / self.cfg["env_resolution_m"])
            if (occupied_area & grid).sum() == 0 and (
                (current_grid > 0) & extended_grid
            ).sum() == 0:
                new_grid = current_grid.copy()
                current_box_index = int(current_box + box_start_index)
                new_grid[grid] = current_box_index
                box["i"] = current_box_index
                box["size"] = [float(v) for v in box_dims[current_box]]
                result_boxes, result_grid = self.place_randomly(
                    rng,
                    angle_range,
                    placed_boxes + [box],
                    box_dims,
                    occupied_area,
                    new_grid,
                    min_dist,
                    box_start_index,
                )
                if len(result_boxes) == len(box_dims):
                    return result_boxes, result_grid

        return placed_boxes, current_grid

    def reset_boxes(self, rng):
        box_dims = []
        for obstacle_spec in self.cfg["obstacle_specification"]:
            for i in range(obstacle_spec[1]):
                if rng.random() < self.cfg["obstacle_placement_probability"]:
                    box_dims += [np.array(self.cfg["box_dim_m"]) * obstacle_spec[0]]

        self.obstacle_boxes, self.obstacle_grid = self.place_randomly(
            rng=rng,
            angle_range=2 * np.pi,
            placed_boxes=[],
            box_dims=box_dims,
            occupied_area=self.expand_occupancy(
                self.border_grid, self.cfg["min_dist_obstacles_border"]
            ),
            current_grid=self.get_base_grid().astype(int),
            min_dist=self.cfg["min_dist_between_obstacles"],
            box_start_index=self.border_grid.max() + 1,
        )

    def reset_sensors_target(self, rng, n_sensors):
        box_dims = [[0.1, 0.1, 0.0]] * (n_sensors + 1)  # add 1 for target
        occupancy = (self.obstacle_grid > 0) | (self.border_grid > 0)
        self.sensor_boxes, self.sensor_grid = self.place_randomly(
            rng=rng,
            angle_range=0.0,
            placed_boxes=[],
            box_dims=box_dims,
            occupied_area=self.expand_occupancy(
                occupancy, self.cfg["min_dist_sensors_obstacles"]
            ),
            current_grid=self.get_base_grid().astype(int),
            min_dist=self.cfg["min_dist_between_sensors"],
            box_start_index=self.obstacle_grid.max() + 1,
        )

    def reset_robot(self, rng):
        box_dims = [[0.35, 0.3, 0.0]]
        occupancy = (
            (self.obstacle_grid > 0) | (self.border_grid > 0) | (self.sensor_grid > 0)
        )
        self.robot_boxes, self.robot_grid = self.place_randomly(
            rng=rng,
            angle_range=0,
            placed_boxes=[],
            box_dims=box_dims,
            occupied_area=self.expand_occupancy(
                occupancy, self.cfg["min_dist_robot_obstacles_sensors"]
            ),
            current_grid=self.get_base_grid().astype(int),
            min_dist=0.0,
            box_start_index=self.sensor_grid.max() + 1,
        )

    def compute_grid(self):
        self.grid = self.border_grid.copy()
        self.grid += self.obstacle_grid
        self.grid += self.sensor_grid
        self.grid += self.robot_grid
