#!/usr/bin/env python3
# wifo_.py
# ROS node: receive I/Q packet, run wifo inference, send back with same [len|timestamp|seq] header,
# plus publish I/Q + seq + timestamp as VizPack msg for visualization.

import socket
import threading
import time
import struct
import argparse

import numpy as np
import rospy
from std_msgs.msg import Header
from wifo_processor.msg import VizPack  # custom message

import model_wifo.trt10 as wifo

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merged node: recv I/Q, wifo infer, send back results'
    )
    parser.add_argument('--in-host',  type=str, default='127.0.0.1',
                        help='Host to bind for incoming data (default: 127.0.0.1)')
    parser.add_argument('--in-port',  type=int, default=8000,
                        help='Port to bind for incoming data (default: 8000)')
    parser.add_argument('--out-host', type=str, default='127.0.0.1',
                        help='Host to bind for outgoing data (default: 127.0.0.1)')
    parser.add_argument('--out-port', type=int, default=8001,
                        help='Port to bind for outgoing data (default: 8001)')
    return parser.parse_args()

class MergedNode:
    def __init__(self, in_host, in_port, out_host, out_port):
        # initialize ROS node
        rospy.init_node('merged_wifo_node', anonymous=False)

        # Publisher for visualization data
        self._iq_pub = rospy.Publisher(
            '/wifo/iq_visual', VizPack, queue_size=10
        )

        # thread synchronization
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._last_ts = None
        self._last_seq = None
        self._last_payload = None

        # incoming socket
        self._in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._in_sock.bind((in_host, in_port))
        self._in_sock.listen(1)
        print(f"[{time.time():.4f}] Listening on {in_host}:{in_port}")
        self._in_conn, _ = self._in_sock.accept()
        print(f"[{time.time():.4f}] Incoming connection established")

        # outgoing socket
        self._out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._out_sock.bind((out_host, out_port))
        self._out_sock.listen(1)
        print(f"[{time.time():.4f}] Listening on {out_host}:{out_port}")
        self._out_conn, _ = self._out_sock.accept()
        print(f"[{time.time():.4f}] Outgoing connection established")

        # start threads
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._proc_loop, daemon=True).start()

    @staticmethod
    def _recv_all(sock, nbytes):
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = sock.recv(nbytes - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def _recv_loop(self):
        """Receive [len|timestamp|seq] header + payload, cache for processing."""
        while not rospy.is_shutdown():
            hdr = self._recv_all(self._in_conn, 4)
            if not hdr:
                break
            payload_len = int.from_bytes(hdr, 'big')

            ts_bytes = self._recv_all(self._in_conn, 8)
            seq_bytes = self._recv_all(self._in_conn, 4)
            t_start = time.time()
            raw = self._recv_all(self._in_conn, payload_len)
            t_end = time.time()

            if raw is None:
                break

            timestamp = struct.unpack('>d', ts_bytes)[0]
            seq = int.from_bytes(seq_bytes, 'big')
            latency = (t_end - t_start) * 1000.0

            with self._lock:
                self._last_ts = timestamp
                self._last_seq = seq
                self._last_payload = raw
            self._event.set()

            print(f"[{t_end:.4f}] Seq={seq}, Step1 recv latency={latency:.3f} ms")

    def _proc_loop(self):
        """On new data, run inference, send back, and publish for RViz."""
        while not rospy.is_shutdown():
            self._event.wait()
            self._event.clear()

            with self._lock:
                timestamp = self._last_ts
                seq = self._last_seq
                raw = self._last_payload

            # Step2: unpack I/Q int16
            t2 = time.time()
            arr = np.frombuffer(raw, dtype=np.int16)  # shape: [2*N]
            wait_latency = (t2 - timestamp) * 1000.0
            print(f"[{t2:.4f}] Seq={seq}, Step2 wait latency={wait_latency:.3f} ms")

            # publish for RViz
            # msg = VizPack()
            # # use original timestamp
            # msg.header = Header(stamp=rospy.Time.from_sec(timestamp), frame_id='wifo_iq')
            # msg.seq = seq
            # msg.data_IQ = arr.tolist()
            # self._iq_pub.publish(msg)

            # Step3: inference
            t3_start = time.time()
            # reshape before model: (1,2,24,4,128)
            input_arr = arr.reshape((1,2,24,4,128)).astype(np.float32)
            result = wifo.model_infer(input_arr)
            t3_end = time.time()
            proc_latency = (t3_end - t3_start) * 1000.0
            print(f"[{t3_end:.4f}] Seq={seq}, Step3 proc latency={proc_latency:.3f} ms")

            # Step4: send back
            t4_start = time.time()
            res_raw = result.astype(np.int16).tobytes()
            length_bytes = len(res_raw).to_bytes(4, 'big')
            ts_bytes = struct.pack('>d', timestamp) # 4-byte uint32
            seq_bytes = struct.pack('>I', seq)
            self._out_conn.sendall(length_bytes + ts_bytes + seq_bytes + res_raw)
            t4_end = time.time()
            send_latency = (t4_end - t4_start) * 1000.0
            print(f"[{t4_end:.4f}] Seq={seq}, Step4 send latency={send_latency:.3f} ms")

    def run(self):
        rospy.spin()
        self._in_conn.close()
        self._out_conn.close()
        self._in_sock.close()
        self._out_sock.close()

if __name__ == '__main__':
    args = parse_args()

    ENGINE_PATH = "asset/wifo_base_int8_trt10.engine"
    BATCH_SIZE  = 1
    INPUT_SHAPE = (BATCH_SIZE, 2, 24, 4, 128)

    wifo.model_init(ENGINE_PATH, INPUT_SHAPE)
    node = MergedNode(args.in_host, args.in_port, args.out_host, args.out_port)
    node.run()
