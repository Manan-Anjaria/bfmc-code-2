import json
import socket
from threading import Thread
# from ..templates.workerprocess import WorkerProcess
from src.templates.workerprocess import WorkerProcess
# from ...templates.workerprocess import WorkerProcess
import time
from typing import Tuple
import cv2
import requests
import numpy as np
import zmq
import joblib

PI_IP = "192.168.152.242"
PORT = 8888
REMOTE_PORT = 8888


def localize(img: np.ndarray) -> Tuple[float, float]:
    AREA_THRES = 100.0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(img, (35, 100, 150), (80, 160, 240))
    cnts = cv2.findContours(
        frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    x = None
    y = None
    if len(cnts) > 0:
        blue_box = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(blue_box) > AREA_THRES:
            # return its center point
            x, y, w, h = cv2.boundingRect(blue_box)
            x = x + w / 2
            y = y + h / 2
    x = round(6 * x / 720, 2) if x else x
    y = round(6 * y / 720, 2) if y else y
    return x, y


def annotate_image(x: float, y: float, image: np.ndarray) -> np.ndarray:
    """Given x and y coordinates of data annotate the image."""
    org = [int((x * 720) / 6), int((y * 720) / 6)]
    if x > 500:
        org[0] = 500
    if y > 650:
        org[1] = 650
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (0, 0, 255)
    thickness = 2

    return cv2.putText(
        image,
        f"car({x},{y})",
        org,
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )


class LocalisationProcess(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        super(LocalisationProcess, self).__init__(inPs, outPs)

    # ===================================== RUN ==========================================
    def run(self, preview=False):
        self.preview = preview
        self.port = PORT
        self.serverIp = PI_IP  # pi addr
        self.threads = list()
        self._init_socket()
        super(LocalisationProcess, self).run()

    # ===================================== INIT SOCKET ==================================
    def _init_socket(self):
        self.port = REMOTE_PORT
        self.serverIp = "0.0.0.0"

        self.server_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        self.server_socket.bind((self.serverIp, self.port))

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        readTh = Thread(
            name="LocSysRecvThread", target=self._read_stream, args=(self.outPs,)
        )
        self.threads.append(readTh)

    # ===================================== READ STREAM ==================================
    def _read_stream(self, outPs):

        # self.server_socket.setblocking(False)
        context_send = zmq.Context()
        pub_loc = context_send.socket(zmq.PUB)
        pub_loc.bind("ipc:///tmp/v31")

        context_recv = zmq.Context()
        sub_loc = context_recv.socket(zmq.SUB)
        sub_loc.setsockopt(zmq.CONFLATE, 1)
        sub_loc.connect("ipc:///tmp/vhl")
        sub_loc.setsockopt_string(zmq.SUBSCRIBE, "")

        count = 0
        skip_count = 24
        r = requests.get(
            "http://10.20.2.114/asp/video.cgi", auth=("admin", "admin"), stream=True
        )
        rx = []
        ry = []
        bytes1 = bytes()
        if r.status_code == 200:
            for idx, chunk in enumerate(r.iter_content(chunk_size=100_000)):
                start_time = time.time()
                count += 1
                bytes1 += chunk
                a = bytes1.find(b"\xff\xd8")  # marks start of the frame
                b = bytes1.find(b"\xff\xd9")  # marks end   of the frame
                # the end of last frame in chunks
                c = bytes1.rfind(b"\xff\xd9")

                if idx < skip_count or a == -1 or b == -1:
                    continue
                jpg = bytes1[a: b + 2]  # get frame based on markers
                bytes1 = bytes1[c + 2:]  # update buffer to store data
                # of last frame present in chunk
                i = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                # specify desired output size
                width = 720
                height = 1280
                input = np.float32(
                    [[38, 344], [617, 38], [1279, 57], [860, 669]])
                output = np.float32(
                    [[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]]
                )
                i = cv2.imdecode(np.frombuffer(
                    jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                # specify desired output size
                width = 720
                # height = 1280
                matrix = cv2.getPerspectiveTransform(input, output)
                image = cv2.warpPerspective(
                    i,
                    matrix,
                    (width, width),
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                x, y = localize(image)
                if x and y:
                    data = {
                        "timestamp": time.time(),
                        "posA": x,
                        "posB": y,
                        "rotA": 0,
                    }
                    data = json.dumps(data).encode()
                    # data = data.decode('utf-8')
                    print(data)
                    imgOutput = annotate_image(x, y, image)
                else:
                    imgOutput = image
                if self.preview:
                    cv2.imshow("Track Image", imgOutput)
                    key = cv2.waitKey(1)
                    if key == 27 or key == 113:
                        # joblib.dump({"x": rx, "y": ry}, "coordlist.z")
                        cv2.destroyAllWindows()
                        break
                    else:
                        if key != -1:
                            print(key)
                        pass
        else:
            print("Received unexpected status code {}".format(r.status_code))
        
        print("------------REACHED BEFORE TRY----------------")
        try:
            print("Starting Home Localization Process")
            while True:
                print("-------------REACHED HERE---------------------")
                # bts, addr = self.server_socket.recvfrom(1024)
                # data = sub_loc.recv()
                print(data)
                # data = data.decode()
                data = json.loads(data)
                pub_loc.send_json(data, flags=zmq.NOBLOCK)

        except Exception as e:
            print("Home LocSys Error")
            print(e)

        finally:
            self.server_socket.close()