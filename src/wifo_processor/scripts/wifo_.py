#!/usr/bin/env python3
# wifo_.py
# ROS node: combined TCP receiver, wifo processor, and TCP sender with minimal latency

import socket
#import pickle  # no longer used for payload
import threading
import time
import argparse

import numpy as np
import rospy

import model_wifo.trt10 as wifo

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merged node: receives (seq, array), runs wifo, sends back results'
    )
    parser.add_argument('--in-host',   type=str, default='127.0.0.1',
                        help='Host to bind for incoming data (default: 127.0.0.1)')
    parser.add_argument('--in-port',   type=int, default=8000,
                        help='Port to bind for incoming data (default: 8000)')
    parser.add_argument('--out-host',  type=str, default='127.0.0.1',
                        help='Host to bind for outgoing data (default: 127.0.0.1)')
    parser.add_argument('--out-port',  type=int, default=8001,
                        help='Port to bind for outgoing data (default: 8001)')
    return parser.parse_args()

class MergedNode:
    def __init__(self, in_host, in_port, out_host, out_port):
        # Initialize ROS node for compatibility
        rospy.init_node('merged_wifo_node', anonymous=False)

        # Shared state and synchronization
        self._lock = threading.Lock()
        self._seq_num = None
        self._array = None
        self._has_new = False
        self._event = threading.Event()
        self._last_recv_ts = None

        # Setup incoming TCP server for pkg_trans.py
        self._in_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._in_sock.bind((in_host, in_port))
        self._in_sock.listen(1)
        print(f"[{time.time():.4f}] Listening for incoming on {in_host}:{in_port}")
        self._in_conn, addr_in = self._in_sock.accept()
        print(f"[{time.time():.4f}] Incoming connection from {addr_in}")

        # Handshake
        self._in_conn.sendall(b'ACK')
        print(f"[{time.time():.4f}] Sent ACK to incoming")

        # Setup outgoing TCP server for pkg_rcv.py
        self._out_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._out_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._out_sock.bind((out_host, out_port))
        self._out_sock.listen(1)
        print(f"[{time.time():.4f}] Listening for outgoing on {out_host}:{out_port}")
        self._out_conn, addr_out = self._out_sock.accept()
        print(f"[{time.time():.4f}] Outgoing connection from {addr_out}")

        # Handshake
        self._out_conn.sendall(b'ACK')
        print(f"[{time.time():.4f}] Sent ACK to outgoing")

        # Start threads for receiving and processing
        threading.Thread(target=self._recv_loop, daemon=True).start()
        threading.Thread(target=self._wifo_loop, daemon=True).start()

    def _recv_loop(self):
        """ Continuously receive (seq, array) and update cache """
        while not rospy.is_shutdown():
            # Read header
            header = self._recv_all(self._in_conn, 4)
            t1_start = time.time()
            if not header:
                print(f"[{time.time():.4f}] incoming closed")
                break
            length = int.from_bytes(header, byteorder='big')

            # Read payload
            payload = self._recv_all(self._in_conn, length)
            if not payload:
                print(f"[{time.time():.4f}] payload truncated")
                break

            # Unpack and cache without pickle
            # Extract sequence number (first 4 bytes)
            seq = int.from_bytes(payload[:4], byteorder='big')
            # Interpret remaining bytes as float32 array
            arr = np.frombuffer(payload[4:], dtype=np.float32)

            t1_end = time.time()
            latency1 = (t1_end - t1_start) * 1000.0

            with self._lock:
                self._seq_num = seq
                self._array = arr
                self._has_new = True
                self._event.set()
                self._last_recv_ts = t1_end

            # Step 1 log
            print(f"[{t1_end:.4f}] Seq={seq}, Step1: recv complete, latency={latency1:.3f} ms")

    def _wifo_loop(self):
        """ Wait for new data, run wifo, send back result (Steps 2-4) """
        while not rospy.is_shutdown():
            # wait for new packet
            self._event.wait()
            self._event.clear()

            # Step 2: measure wait latency
            t2_start = time.time()
            with self._lock:
                seq = self._seq_num
                array = self._array
                recv_ts = self._last_recv_ts
                self._has_new = False
            
            wait_latency = (t2_start - recv_ts) * 1000.0
            print(f"[{t2_start:.4f}] Seq={seq}, Step2: cache, wait_latency={wait_latency:.3f} ms")

            # Step 3: wifo processing (simulation)
            t3_start = time.time()
            # result = array  # no-op processing
            result = wifo.model_infer(array)

            t3_end = time.time()
            proc_latency = (t3_end - t3_start) * 1000.0
            print(f"[{t3_end:.4f}] Seq={seq}, Step3: processing done, latency={proc_latency:.3f} ms")

            # Step 4: send back without pickle
            t4_start = time.time()
            # Serialize seq + raw data
            seq_bytes = seq.to_bytes(4, byteorder='big')
            data = seq_bytes + result.tobytes()
            header = len(data).to_bytes(4, byteorder='big')
            self._out_conn.sendall(header + data)
            t4_end = time.time()
            send_latency = (t4_end - t4_start) * 1000.0
            print(f"[{t4_end:.4f}] Seq={seq}, Step4: send complete, latency={send_latency:.3f} ms")

    @staticmethod
    def _recv_all(sock, nbytes):
        """ Helper: recv exactly nbytes or return None if EOF """
        buf = bytearray()
        while len(buf) < nbytes:
            chunk = sock.recv(nbytes - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def run(self):
        rospy.spin()
        # cleanup
        self._in_conn.close()
        self._out_conn.close()
        self._in_sock.close()
        self._out_sock.close()
        print(f"[{time.time():.4f}] Merged node terminated")

if __name__ == '__main__':
    args = parse_args()

    import os

    print("Current Dir:", os.getcwd())
    ENGINE_PATH  = "asset/wifo_base_int8_trt10.engine"
    BATCH_SIZE   = 1
    INPUT_SHAPE  = (BATCH_SIZE, 2, 24, 4, 128)
    

    wifo.model_init(ENGINE_PATH, INPUT_SHAPE)

    node = MergedNode(args.in_host, args.in_port,
                      args.out_host, args.out_port)
    node.run()
