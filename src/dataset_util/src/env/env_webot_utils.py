import socket
from io import BytesIO
import numpy as np


def query_picture_from_sensor(port, socket_buffer_size=1024):
    try:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        conn.connect(("127.0.0.1", port))
    except ConnectionRefusedError:
        print("ERR")
        return None

    length = None
    frameBuffer = bytearray()
    while True:
        data = conn.recv(socket_buffer_size)
        frameBuffer += data
        if len(frameBuffer) == length:
            break
        while True:
            if length is None:
                if b":" not in frameBuffer:
                    break
                # remove the length bytes from the front of frameBuffer
                # leave any remaining bytes in the frameBuffer!
                length_str, ignored, frameBuffer = frameBuffer.partition(b":")
                length = int(length_str)
            if len(frameBuffer) < length:
                break
            # split off the full message from the remaining bytes
            # leave any remaining bytes in the frameBuffer!
            frameBuffer = frameBuffer[length:]
            length = None
            break

    return np.load(BytesIO(frameBuffer))["frame"]


def rm_if_exists(robot, name):
    node = robot.getFromDef(name)
    if node is not None:
        node.remove()


def reset_boxes(robot, obstacle_boxes):
    for i in range(100):
        rm_if_exists(robot, f"box_{i}")

    for i, t in enumerate(obstacle_boxes):
        translate = " ".join([str(v) for v in t["t"]])
        rotate = " ".join([str(v) for v in t["r"]])
        box_size = " ".join([str(v) for v in t["size"]])
        robot.getFromDef("obstacles").getField("children").importMFNodeFromString(
            -1,
            f'DEF box_{i} CardboardBox {{ name "box_{i}" size {box_size} translation {translate} rotation {rotate}}}',
        )


def reset_sensors_target(robot, sensor_boxes, target_box, robot_box):
    for i, t in enumerate(sensor_boxes):
        sensor = robot.getFromDef(f"sensor_{i + 1}")
        sensor.getField("translation").setSFVec3f(t["t"])
        sensor.getField("rotation").setSFRotation(t["r"])

    target = robot.getFromDef("target")
    target.getField("translation").setSFVec3f(target_box["t"])
    target.getField("rotation").setSFRotation(target_box["r"])

    robomaster = robot.getFromDef("robot")
    robomaster.getField("translation").setSFVec3f(robot_box["t"])
    robomaster.getField("rotation").setSFRotation(robot_box["r"])


def initialize_scene(robot, n_sensors, port_start=8000):
    environment_children = robot.getFromDef("environment").getField("children")

    rm_if_exists(robot, "robot")
    environment_children.importMFNodeFromString(
        -1,
        f'DEF robot RoboMaster {{ port "{port_start}" name "sensor_raw_0" }}',
    )

    for i in range(1, n_sensors + 1):
        rm_if_exists(robot, f"sensor_{i}")
        environment_children.importMFNodeFromString(
            -1,
            f'DEF sensor_{i} Sensor {{ port "{port_start + i}" name "sensor_raw_{i}" }}',
        )

    rm_if_exists(robot, "target")
    environment_children.importMFNodeFromString(-1, f"DEF target Target {{ }}")
