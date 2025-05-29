#!/usr/bin/env python3
# pkg_trans.py
# Python client module: connects to server A, generates random arrays and sends them every second

import socket
import time
# import pickle  # no longer used for payload
# import torch
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='TCP client for sending sequence number and random array')
    parser.add_argument('--host', '-H', type=str, default='127.0.0.1',
                        help='Server host to connect to (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8000,
                        help='Server port to connect to (default: 8000)')
    return parser.parse_args()

def pkg_trans(server_host: str, server_port: int):
    """
    Connects to server A over TCP and sends (sequence_number, array) tuples every second.
    Each transmission logs:
      Step1: array generation timestamp and generation latency
      Step2: send completion timestamp and send-call latency
    """
    # Create TCP socket and connect
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[{time.time():.4f}] Connected to {server_host}:{server_port}")

    seq_num = 1  # Initialize sequence number
    try:
        while True:
            # Step1: generate the random array and measure generation time

            # array = torch.rand(10240)  # generate 1×10240 random tensor
            array = np.random.rand(24576).astype(np.float32)
            t1_start = time.time()

            # Package with native bytes (4-byte seq + raw data)
            # Convert tensor to numpy bytes

            # raw_array = array.numpy().tobytes()
            raw_array = array.tobytes()

            # Serialize sequence number as 4-byte big-endian
            seq_bytes = seq_num.to_bytes(4, byteorder='big')
            # Combined payload
            data = seq_bytes + raw_array
            # Prepend 4-byte length header
            header = len(data).to_bytes(4, byteorder='big')

            t1_end = time.time()
            gen_latency = (t1_end - t1_start) * 1000.0
            print(f"[{t1_end:.4f}] Seq={seq_num}, Step1: array seq generated, gen_latency={gen_latency:.3f} ms")

            # Step2: send the packet and measure send-call latency
            t2_start = time.time()
            sock.sendall(header + data)
            t2_end = time.time()
            send_latency = (t2_end - t2_start) * 1000.0
            total_bytes = len(header) + len(data)
            print(f"[{t2_end:.4f}] Seq={seq_num}, Step2: send complete, send_latency={send_latency:.3f} ms, "
                  f"sent {total_bytes} bytes")

            seq_num += 1  # Increment sequence number

            # Wait one second
            time.sleep(1.0)

    except KeyboardInterrupt:
        # Graceful shutdown on interrupt
        print(f"[{time.time():.4f}] Shutting down pkg_trans client")
    finally:
        sock.close()

if __name__ == '__main__':
    args = parse_args()
    pkg_trans(args.host, args.port)
