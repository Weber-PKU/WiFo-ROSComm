#!/usr/bin/env python3
# pkg_rcv.py
# TCP client: receive processed I/Q back from wifo_.py, unpack [len|timestamp|seq], log latencies.

import socket
import time
import struct
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='TCP client receiving I/Q from wifo server'
    )
    parser.add_argument('--host', '-H', type=str, default='127.0.0.1',
                        help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8001,
                        help='Server port (default: 8001)')
    return parser.parse_args()

def recv_all(sock, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        packet = sock.recv(nbytes - len(buf))
        if not packet:
            return None
        buf.extend(packet)
    return bytes(buf)

def pkg_rcv(server_host: str, server_port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[{time.time():.4f}] Connected to {server_host}:{server_port}")

    try:
        while True:
            hdr = recv_all(sock, 4)
            if not hdr:
                break
            payload_len = int.from_bytes(hdr, 'big')

            ts_bytes = recv_all(sock, 8)
            seq_bytes = recv_all(sock, 2)
            t1_start = time.time()
            raw = recv_all(sock, payload_len)
            t1_end = time.time()
            if raw is None:
                break

            timestamp = struct.unpack('>d', ts_bytes)[0]
            seq = int.from_bytes(seq_bytes, 'big')
            latency1 = (t1_end - t1_start) * 1000.0
            print(f"[{t1_end:.4f}] Seq={seq}, Step1 recv latency={latency1:.3f} ms, timestamp={timestamp:.6f}")

            # Step2: unpack I/Q
            t2_start = time.time()
            arr = np.frombuffer(raw, dtype=np.int16)
            t2_end = time.time()
            latency2 = (t2_end - t2_start) * 1000.0
            print(f"[{t2_end:.4f}] Seq={seq}, Step2 unpack latency={latency2:.3f} ms, samples={len(arr)//2}")

    except KeyboardInterrupt:
        print(f"[{time.time():.4f}] Shutting down pkg_rcv")
    finally:
        sock.close()

if __name__ == '__main__':
    args = parse_args()
    pkg_rcv(args.host, args.port)
