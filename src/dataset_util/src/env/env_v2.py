import numpy as np
import scipy.ndimage
import cv2
import pyvisgraph as vg
import shapely
from shapely.geometry import Point, MultiPolygon, Polygon, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt


def rand(rng, a: float, b: float, size=None):
    return (a - b) * rng.random(size) + b


def plot_polygon(polygon, ax, **kwargs):
    if isinstance(polygon, Polygon):
        polygon = MultiPolygon([polygon])
    assert isinstance(polygon, MultiPolygon)

    for p in polygon.geoms:
        ax.plot(*p.exterior.xy, **kwargs)
        for interior in p.interiors:
            ax.plot(*interior.xy, **kwargs)


class Env:
    def __init__(self, cfg):
        self.cfg = cfg

    def generate(self, rng):
        self.border_boxes = self.compute_border_boxes()

        box_dims = []
        for obstacle_spec in self.cfg["obstacle_specification"]:
            for i in range(obstacle_spec[1]):
                if rng.random() < self.cfg["obstacle_placement_probability"]:
                    box_dims += [np.array(self.cfg["box_dim_m"]) * obstacle_spec[0]]

        border_occupied_area = shapely.ops.unary_union(
            [box["shape"] for box in self.border_boxes]
        )
        self.obstacle_boxes, _ = self.place_randomly(
            rng=rng,
            box_dims=box_dims,
            occupied_area=border_occupied_area,
            min_dist_between_boxes=self.cfg["min_dist_between_obstacles"],
            min_dist_to_occupied_area=self.cfg["min_dist_obstacles_border"],
        )
        if len(self.obstacle_boxes) == 0:
            return False

        sensor_occupied_area = shapely.ops.unary_union(
            [border_occupied_area] + [box["shape"] for box in self.obstacle_boxes]
        )

        sensor_target_boxes, _ = self.place_randomly(
            rng=rng,
            box_dims=[[0.1, 0.1, 0.0]] * (self.cfg["n_sensors"] + 1),
            angle_range=0.0,
            occupied_area=sensor_occupied_area,  # buffer for min dist to obstacles
            min_dist_between_boxes=self.cfg["min_dist_between_sensors"],
            min_dist_to_occupied_area=self.cfg["min_dist_sensors_obstacles"],
            # navigatable_area_clearance=self.cfg["obstacle_clearance_sensor_robot"],
            # navigatable_box_index=0,
        )
        if len(sensor_target_boxes) < (self.cfg["n_sensors"] + 1):
            return False

        self.sensor_boxes = sensor_target_boxes[1:]
        self.target_boxes = sensor_target_boxes[:1]

        robot_occupied_area = shapely.ops.unary_union(
            [sensor_occupied_area] + [box["shape"] for box in self.sensor_boxes]
        )
        robot_boxes, _ = self.place_randomly(
            rng=rng,
            angle_range=0.0,
            placed_boxes=self.target_boxes,
            box_dims=[[0.35, 0.35, 0.0]],
            occupied_area=robot_occupied_area,
            current_placement=MultiPolygon([self.target_boxes[0]["shape"]]),
            min_dist_between_boxes=self.cfg["min_dist_between_robot_target"],
            min_dist_to_occupied_area=self.cfg["min_dist_robot_obstacles_sensors"],
            # navigatable_area_clearance=self.cfg["obstacle_clearance_sensor_robot"],
            # navigatable_box_index=1,
        )
        if len(robot_boxes) < len(self.target_boxes) + 1:
            return False

        self.robot_boxes = robot_boxes[1:]

        self.sensor_paths = self.compute_paths(
            self.target_boxes[0],
            self.sensor_boxes,
            sensor_occupied_area,
            self.cfg["obstacle_clearance_sensor_robot"],
            [[0.0, 0.0]],
        )

        self.robot_paths = self.compute_paths(
            self.target_boxes[0],
            self.robot_boxes,
            robot_occupied_area,
            self.cfg["obstacle_clearance_sensor_robot"],
            [[0.0, 0.0]],
        )

        if any([len(p[0]["path"]) == 0 for p in self.robot_paths + self.sensor_paths]):
            # no path for any of the boxes
            return False

        return True

    def get_box_polygon(self, pos, angle, dim):
        box = shapely.geometry.box(-dim[0] / 2, -dim[1] / 2, dim[0] / 2, dim[1] / 2)
        box_rot = shapely.affinity.rotate(box, angle, use_radians=True)
        box_trans = shapely.affinity.translate(box_rot, pos[0], pos[1])
        assert box_trans.is_valid
        return {
            "meta": {
                "t": [float(pos[0]), float(pos[1]), float(dim[2] / 2)],
                "r": [0.0, 0.0, 1.0, float(angle)],
                "size": [float(d) for d in dim],
            },
            "shape": box_trans,
        }

    def compute_border_boxes(self):
        boxes = []

        offset_a = (
            (np.array(self.cfg["env_size_boxes"]) - 1) / 2 * self.cfg["box_dim_m"][0]
        )
        offset_b = (
            np.array(self.cfg["env_size_boxes"]) / 2 * self.cfg["box_dim_m"][0]
            + self.cfg["box_dim_m"][1] / 2
        )

        for i in range(self.cfg["env_size_boxes"][0]):
            boxes.append(
                self.get_box_polygon(
                    [
                        self.cfg["box_dim_m"][0] * i - offset_a[0],
                        offset_b[1] - self.cfg["box_dim_m"][1],
                    ],
                    0.0,
                    self.cfg["box_dim_m"],
                )
            )

        for i in range(self.cfg["env_size_boxes"][1]):
            boxes.append(
                self.get_box_polygon(
                    [
                        offset_b[0],
                        self.cfg["box_dim_m"][0] * i - offset_a[1],
                    ],
                    1.57,
                    self.cfg["box_dim_m"],
                )
            )

        for i in range(self.cfg["env_size_boxes"][0]):
            boxes.append(
                self.get_box_polygon(
                    [
                        self.cfg["box_dim_m"][0] * i
                        - offset_a[0]
                        + self.cfg["box_dim_m"][1],
                        -offset_b[1],
                    ],
                    0.0,
                    self.cfg["box_dim_m"],
                )
            )

        for i in range(self.cfg["env_size_boxes"][1]):
            boxes.append(
                self.get_box_polygon(
                    [
                        -offset_b[0] + self.cfg["box_dim_m"][1],
                        self.cfg["box_dim_m"][0] * i
                        - offset_a[1]
                        - self.cfg["box_dim_m"][1],
                    ],
                    1.57,
                    self.cfg["box_dim_m"],
                )
            )
        return boxes

    def compute_visgraph_obstacles(self, obstacle_polygons):
        # buffer slightly since due to the simplification and check for intersection we might actually
        # intersect the raw obstacle polygon
        if isinstance(obstacle_polygons, Polygon):
            # Make a multiPolygon as it could sometimes be a Polygon but we want to be able to iterate
            obstacle_polygons = MultiPolygon([obstacle_polygons])
        processed_obstacle_polygons = []
        for polygon in obstacle_polygons.geoms:
            polygon_simple = polygon.simplify(0.025)
            polygon_exterior_coords = np.array(polygon_simple.exterior.coords)
            interiors_points = {}
            for interior in polygon_simple.interiors:
                interior_coords = np.array(interior.coords)
                # It could be the case that we have holes in a polygon after buffering and merging.
                # Holes cannot be handled by pyvisgraph, so we have to connect them to the exterior.
                p_interior, p_exterior = shapely.ops.nearest_points(
                    interior, polygon_simple.exterior
                )
                p_ext_idx = np.linalg.norm(
                    polygon_exterior_coords - p_exterior, axis=1
                ).argmin()
                p_int_idx = np.linalg.norm(
                    interior_coords - p_interior, axis=1
                ).argmin()
                # roll the interior points so that it starts with the point closest to the exterior
                interiors_points[int(p_ext_idx)] = np.roll(
                    interior_coords, -p_int_idx, axis=0
                )
            points = []
            p_ext_idx = 0  # initialize in case interiors_points is empty
            for p_ext_idx, interior_points in interiors_points.items():
                points += [
                    vg.Point(*c) for c in polygon_exterior_coords[: p_ext_idx + 1]
                ]
                points += [vg.Point(*c) for c in interior_points]
                points += [
                    vg.Point(interior_points[0][0], interior_points[0][1])
                ]  # add the first point again
                points += [
                    vg.Point(*polygon_exterior_coords[p_ext_idx][:2])
                ]  # add the first point again
            points += [vg.Point(*c) for c in polygon_exterior_coords[p_ext_idx:]]
            processed_obstacle_polygons.append(points)
        return processed_obstacle_polygons

    def compute_paths(
        self, goal_box, other_boxes, occupied_area, navigatable_area_clearance, delta_ps
    ):
        paths = []
        for i in range(len(other_boxes)):
            subpaths = []
            for delta_p in np.array(delta_ps):
                start_box = other_boxes[i]
                obstacle_boxes = other_boxes[:i] + other_boxes[i + 1 :]

                raw_obstacle_polygons = unary_union(
                    [box["shape"] for box in obstacle_boxes] + [occupied_area]
                )
                buffered_obstacle_polygons = raw_obstacle_polygons.buffer(
                    navigatable_area_clearance
                )
                obstacle_polygons = self.compute_visgraph_obstacles(
                    buffered_obstacle_polygons
                )

                visibility_graph = vg.VisGraph()
                try:
                    visibility_graph.build(obstacle_polygons, status=False)
                    p_start = start_box["meta"]["t"][:2] + delta_p
                    if not buffered_obstacle_polygons.simplify(0.025).contains(
                        Point(p_start[0], p_start[1])
                    ):
                        start_point = vg.Point(*(p_start))
                        end_point = vg.Point(
                            goal_box["meta"]["t"][0], goal_box["meta"]["t"][1]
                        )

                        path = visibility_graph.shortest_path(start_point, end_point)
                        path_array = [(path.x, path.y) for path in path]
                        assert len(path_array) >= 2
                    else:
                        path_array = []
                except KeyError:
                    path_array = []
                except UnboundLocalError:
                    path_array = []
                except ZeroDivisionError:
                    path_array = []

                if LineString(path_array).intersects(raw_obstacle_polygons):
                    path_array = []
                subpaths.append({"delta_p": delta_p.tolist(), "path": path_array})
            paths.append(subpaths)
        return paths

    def place_randomly(
        self,
        rng,
        box_dims,
        angle_range=2 * np.pi,
        placed_boxes=[],
        paths=[],
        occupied_area=MultiPolygon([]),
        current_placement=MultiPolygon([]),
        min_dist_between_boxes=0.1,
        min_dist_to_occupied_area=0.1,
        navigatable_box_index=-1,
        navigatable_area_clearance=0.0,
    ):
        if len(box_dims) == 0:
            return placed_boxes, paths

        for i in range(100):
            rand_angle = rand(rng, 0, angle_range)
            world_min_x, world_min_y, world_max_x, world_max_y = occupied_area.bounds
            rand_pos = [
                rand(rng, world_min_x, world_max_x),
                rand(rng, world_min_y, world_max_y),
            ]
            box = self.get_box_polygon(rand_pos, rand_angle, box_dims[0])
            expanded_box = box["shape"].buffer(min_dist_between_boxes).simplify(0.025)
            expanded_occupied_area = occupied_area.buffer(
                min_dist_to_occupied_area
            ).simplify(0.025)

            if not expanded_box.intersects(current_placement) and not box[
                "shape"
            ].intersects(expanded_occupied_area):
                new_placed_boxes = placed_boxes + [box]
                if navigatable_box_index >= 0 and len(new_placed_boxes) >= 2:
                    goal_box = new_placed_boxes[navigatable_box_index]
                    other_boxes = (
                        new_placed_boxes[:navigatable_box_index]
                        + new_placed_boxes[navigatable_box_index + 1 :]
                    )
                    paths = self.compute_paths(
                        goal_box,
                        other_boxes,
                        occupied_area,
                        navigatable_area_clearance,
                        [[0.0, 0.0]],
                    )
                    if any([len(p[0]["path"]) == 0 for p in paths]):
                        # Could not compute a path for any of the boxes
                        continue

                result_boxes, paths = self.place_randomly(
                    rng,
                    box_dims[1:],
                    angle_range,
                    new_placed_boxes,
                    paths,
                    occupied_area,
                    current_placement.union(box["shape"]),
                    min_dist_between_boxes,
                    min_dist_to_occupied_area,
                    navigatable_box_index,
                    navigatable_area_clearance,
                )
                if len(result_boxes) == len(placed_boxes) + len(box_dims):
                    return result_boxes, paths

        return placed_boxes, paths

    def render(
        self,
        ax,
        robot=True,
        obstacles=True,
        sensors=True,
        target=True,
        sensor_paths=True,
        robot_paths=True,
        render_only_direct_path=False,
    ):
        ax.set_aspect(1.0)
        if sensor_paths:
            for path in self.sensor_paths:
                for subpath in path:
                    path_array = np.array(subpath["path"])
                    if len(path_array) > 0:
                        ax.plot(
                            path_array[:, 0],
                            path_array[:, 1],
                            color="blue",
                            linewidth=2,
                        )
                    if render_only_direct_path:
                        break
        if robot_paths:
            for path in self.robot_paths:
                for subpath in path:
                    path_array = np.array(subpath["path"])
                    if len(path_array) > 0:
                        ax.plot(
                            path_array[:, 0], path_array[:, 1], color="red", linewidth=4
                        )
                    if render_only_direct_path:
                        break

        if obstacles:
            for box in self.border_boxes + self.obstacle_boxes:
                plot_polygon(box["shape"], ax, color="black")
        if sensors:
            for box in self.sensor_boxes:
                plot_polygon(box["shape"], ax, color="blue", linewidth=6)
        if robot:
            for box in self.robot_boxes:
                plot_polygon(box["shape"], ax, color="red")
        if target:
            for box in self.target_boxes:
                plot_polygon(box["shape"], ax, color="green", linewidth=6)

    def save_state(self):
        return {
            "cfg": self.cfg,
            "border_obstacles": [box["meta"] for box in self.border_boxes],
            "inner_obstacles": [box["meta"] for box in self.obstacle_boxes],
            "sensors": [box["meta"] for box in self.sensor_boxes],
            "targets": [box["meta"] for box in self.target_boxes],
            "robots": [box["meta"] for box in self.robot_boxes],
            "sensor_paths": self.sensor_paths,
            "robot_paths": self.robot_paths,
        }

    def restore_state(self, state):
        def restore_polygons(boxes_meta):
            return [
                self.get_box_polygon(meta["t"][:2], meta["r"][3], meta["size"])
                for meta in boxes_meta
            ]

        self.cfg = state["cfg"]
        self.robot_paths = state["robot_paths"]
        self.sensor_paths = state["sensor_paths"]
        self.border_boxes = restore_polygons(state["border_obstacles"])
        self.obstacle_boxes = restore_polygons(state["inner_obstacles"])
        self.sensor_boxes = restore_polygons(state["sensors"])
        self.target_boxes = restore_polygons(state["targets"])
        self.robot_boxes = restore_polygons(state["robots"])


env_sample_cfg = {
    "n_sensors": 7,
    "box_dim_m": [0.7, 0.5, 0.5],
    "env_size_boxes": [7, 9],
    "env_resolution_m": 0.01,
    "min_dist_obstacles_border": 0.05,
    "min_dist_between_obstacles": 0.05,
    "min_dist_sensors_obstacles": 0.26,
    "min_dist_between_sensors": 1.0,
    "min_dist_between_robot_target": 0.1,
    "min_dist_robot_obstacles_sensors": 0.26,
    "obstacle_placement_probability": 0.8,
    "obstacle_clearance_sensor_robot": 0.25,
    "obstacle_specification": [
        # box factor and number of boxes to place
        ([3, 1, 1], 1),
        ([2, 1, 1], 2),
        ([1, 2, 1], 1),
        ([1, 1, 1], 2),
    ],
}


def test_save_load():
    env = Env(env_sample_cfg)
    rng = np.random.default_rng(0)
    env.generate(rng)
    meta = env.save_state()

    env_load = Env(meta["cfg"])
    env_load.restore_state(meta)
    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(111)
    env_load.render(ax)
    plt.show()


def test_render():
    env = Env(env_sample_cfg)
    for seed in range(100):
        rng = np.random.default_rng(seed)
        env.generate(rng)

        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        env.render(ax)
        plt.show()


if __name__ == "__main__":
    test_save_load()
