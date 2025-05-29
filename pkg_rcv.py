#!/usr/bin/env python3
# pkg_rcv.py
# Python client module: connects to server B, receives (seq, array) tuples,
# unpacks into seq and tensor, and logs receive & unpack latencies.

import socket
import time
# import pickle  # no longer used
# import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='TCP client for receiving seq and array from pkg_trans_ros'
    )
    parser.add_argument('--host', '-H', type=str, default='127.0.0.1',
                        help='Server host to connect to (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8001,
                        help='Server port to connect to (default: 8001)')
    return parser.parse_args()

def recv_all(sock, length):
    """
    English comment: receive exactly `length` bytes or return None if connection closed.
    """
    buf = bytearray()
    while len(buf) < length:
        packet = sock.recv(length - len(buf))
        if not packet:
            return None
        buf.extend(packet)
    return bytes(buf)

def pkg_rcv(server_host: str, server_port: int):
    """
    Connects to server B over TCP, performs handshake, then continuously
    receives (seq, array) tuples, unpacks into seq and tensor, and logs:
      - Step1: receive latency (header+payload)
      - Step2: unpack & tensor conversion latency
    """
    # create and connect socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[{time.time():.4f}] Connected to {server_host}:{server_port}")

    # handshake: expect 'ACK'
    ack = sock.recv(3)
    if ack == b'ACK':
        print(f"[{time.time():.4f}] Handshake successful: received ACK")
    else:
        print(f"[{time.time():.4f}] Handshake warning: expected ACK but got {ack!r}")

    try:
        while True:
            # Step1: receive header+payload
            header = recv_all(sock, 4)
            if header is None:
                print(f"[{time.time():.4f}] Step1: connection closed")
                break
            length = int.from_bytes(header, byteorder='big')
            
            t1_start = time.time()
            payload = recv_all(sock, length)
            t1_end = time.time()
            if payload is None:
                print(f"[{time.time():.4f}] Step1: payload truncated")
                break

            latency1 = (t1_end - t1_start) * 1000.0
            print(f"[{t1_end:.4f}] Seq=Unknown, Step1: recv complete, latency={latency1:.3f} ms")

            # Step2: unpack without pickle
            t2_start = time.time()
            # Extract sequence number
            seq = int.from_bytes(payload[:4], byteorder='big')
            # Convert remaining bytes directly to tensor
            # buf = bytearray(payload[4:])               # make writable buffer
            # tensor = torch.frombuffer(buf, dtype=torch.float32)
            
            buf = payload[4:]
            array = np.frombuffer(buf, dtype=np.float32)

            t2_end = time.time()
            latency2 = (t2_end - t2_start) * 1000.0

            print(f"[{t2_end:.4f}] Seq={seq}, Step2: unpack complete, latency={latency2:.3f} ms")

    except KeyboardInterrupt:
        print(f"[{time.time():.4f}] Shutting down pkg_rcv client")
    finally:
        sock.close()

if __name__ == '__main__':
    args = parse_args()
    pkg_rcv(args.host, args.port)
