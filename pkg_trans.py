#!/usr/bin/env python3
# pkg_trans.py
# TCP client: generate random I/Q pairs, package with [len|timestamp|seq] header, send every second.

import socket
import time
import struct
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TCP client sending I/Q data with timestamp header')
    parser.add_argument('--host', '-H', type=str, default='127.0.0.1',
                        help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8000,
                        help='Server port (default: 8000)')
    return parser.parse_args()

def pkg_trans(server_host: str, server_port: int):
    """
    Connects to server A, generates 12288 I/Q int16 samples,
    packages [4B len][8B timestamp][2B seq][payload], sends every second.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[{time.time():.4f}] Connected to {server_host}:{server_port}")

    seq_num = 0
    N = 24576 // 2  # number of I/Q pairs

    try:
        while True:
            # generate random I/Q int16 array
            I = np.random.randint(-32768, 32767, size=N, dtype=np.int16)
            Q = np.random.randint(-32768, 32767, size=N, dtype=np.int16)
            array_IQ = np.empty(2 * N, dtype=np.int16)
            array_IQ[0::2] = I
            array_IQ[1::2] = Q
            raw_array = array_IQ.tobytes()

            # header fields
            timestamp = time.time()
            ts_bytes = struct.pack('>d', timestamp)      # 8-byte big-endian double
            seq_bytes = struct.pack('>H', seq_num & 0xFFFF)  # 2-byte big-endian uint16
            length_bytes = len(raw_array).to_bytes(4, byteorder='big')

            # send in one shot
            packet = length_bytes + ts_bytes + seq_bytes + raw_array
            t_start = time.time()
            sock.sendall(packet)
            t_end = time.time()

            print(f"[{t_end:.4f}] Seq={seq_num}, sent {len(packet)} bytes, "
                  f"encode_latency={(t_end-t_start)*1000:.3f} ms")

            seq_num += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print(f"[{time.time():.4f}] Shutting down pkg_trans")
    finally:
        sock.close()

if __name__ == '__main__':
    args = parse_args()
    pkg_trans(args.host, args.port)