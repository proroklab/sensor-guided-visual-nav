"""sensor controller."""

from controller import Robot
import sys
import socket
import numpy as np
from io import BytesIO
import threading


class StreamerThread(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.conn.bind(("127.0.0.1", port))
        self.conn.listen(1)
        self.frame = None

    def update_frame(self, frame):
        f = BytesIO()
        np.savez(f, frame=frame)

        packet_size = len(f.getvalue())
        header = "{0}:".format(packet_size)
        header = bytes(header.encode())  # prepend length of array

        out = bytearray()
        out += header

        f.seek(0)
        out += f.read()
        self.frame = out

    def run(self):
        while True:
            if self.frame is None:
                continue
            client_connection, client_address = self.conn.accept()
            client_connection.sendall(self.frame)
            client_connection.close()


robot = Robot()

timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera0")
camera.enable(10)

assert len(sys.argv) >= 2

port = int(sys.argv[1])

streamer_thread = StreamerThread(port)
streamer_thread.start()
print("Sensor start")
while robot.step(timestep) != -1:
    image_raw = np.frombuffer(camera.getImage(), dtype=np.uint8)
    image = image_raw.reshape(camera.getWidth(), camera.getHeight(), 4)[:, :, :3]
    streamer_thread.update_frame(image)
streamer_thread.join()
