import json
import socket
from threading import Thread
from src.templates.workerprocess import WorkerProcess
import time
import zmq
import numpy as np
import cv2
from typing import Tuple

REMOTE_PORT = 8888

class LocalisationProcess(WorkerProcess):
    # ===================================== INIT =========================================
    def __init__(self, inPs, outPs):
        """Run on raspberry. It forwards the control messages received from socket to the serial handler

        Parameters
        ------------
        inPs : list(Pipe)
            List of input pipes (not used at the moment)
        outPs : list(Pipe)
            List of output pipes (order does not matter)
        """

        super(LocalisationProcess, self).__init__(inPs, outPs)

    # ===================================== RUN ==========================================
    def run(self):
        """Apply the initializing methods and start the threads"""
        self._init_socket()
        super(LocalisationProcess, self).run()

    # ===================================== INIT SOCKET ==================================
    def _init_socket(self):
        """Initialize the communication socket server."""
        self.port = REMOTE_PORT
        # self.serverIp = "192.168.152.242"
        self.serverIp = "0.0.0.0"

        self.server_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )
        self.server_socket.bind((self.serverIp, self.port))

    # ===================================== INIT THREADS =================================
    def _init_threads(self):
        """Initialize the read thread to transmite the received messages to other processes."""
        readTh = Thread(
            name="LocSysRecvThread", target=self._read_stream, args=(self.outPs,)
        )
        self.threads.append(readTh)

    # ===================================== READ STREAM ==================================
    def _read_stream(self, outPs):
        """Receive the message and forwards them to the SerialHandlerProcess.

        Parameters
        ----------
        outPs : list(Pipe)
            List of the output pipes.
        """
        # self.server_socket.setblocking(False)
        context_send = zmq.Context()
        pub_loc = context_send.socket(zmq.PUB)
        pub_loc.bind("ipc:///tmp/v31")
        
        context_recv = zmq.Context()
        sub_loc = context_recv.socket(zmq.SUB)
        sub_loc.setsockopt(zmq.CONFLATE, 1)
        sub_loc.connect("ipc:///tmp/vhl")
        sub_loc.setsockopt_string(zmq.SUBSCRIBE, "")

        print("------------REACHED BEFORE TRY----------------")

        try:
            print("Starting Home Localization Process")
            while True:
                print("-------------REACHED HERE---------------------")
                bts, addr = self.server_socket.recvfrom(1024)
                print(addr)
                # data = sub_loc.recv()
                # data = data.decode()
                data = bts.decode()
                print(data)
                data = json.loads(data)
                pub_loc.send_json(data, flags=zmq.NOBLOCK)
        except Exception as e:
            print("Home LocSys Error")
            print(e)

        finally:
            self.server_socket.close()