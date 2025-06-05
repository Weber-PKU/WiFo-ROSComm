#!/usr/bin/env python3
# pkg_trans.py
# TCP client: generate I/Q sine waves with continuous frequency jitter, package with [len|timestamp|seq] header, send every second.

import socket
import time
import struct
import numpy as np
import argparse

class ToneGenerator:
    """
    Generate IQ samples consisting of two superposed tones with continuous frequency jitter.
    """

    def __init__(self,
                 center_freqs=(1000.0, 2000.0),
                 jitter_range=50.0,
                 jitter_step=1.0,
                 sample_rate=12288):
        """
        :param center_freqs: tuple of two center frequencies in Hz
        :param jitter_range: maximum deviation from center frequency in Hz
        :param jitter_step: maximum change per packet in Hz (random walk step)
        :param sample_rate: sampling rate in samples per second
        """
        self.center_freqs = np.array(center_freqs, dtype=float)
        self.jitter_range = jitter_range
        self.jitter_step = jitter_step
        self.fs = sample_rate
        # start at center frequencies
        self.freqs = self.center_freqs.copy()
        # initial phases for continuity
        self.phases = np.zeros(2, dtype=float)
        # amplitude scaling to int16
        self.amp = 0.4 * np.iinfo(np.int16).max

    def generate(self, N):
        """
        生成 N 对 I/Q 样本并返回 bytes 格式
        :param N: number of I/Q pairs
        :return: raw bytes of interleaved int16 [I0,Q0,I1,Q1,...]
        """
        # 1. 频率随机游走
        delta = np.random.uniform(-self.jitter_step, self.jitter_step, size=2)
        self.freqs += delta
        # 限制在中心频率 ± 抖动范围内
        min_f = self.center_freqs - self.jitter_range
        max_f = self.center_freqs + self.jitter_range
        self.freqs = np.clip(self.freqs, min_f, max_f)

        # 2. 时间向量
        t = np.arange(N) / self.fs

        # 3. 相位计算（累积相位 + 新一包相位增量）
        phi1 = self.phases[0] + 2 * np.pi * self.freqs[0] * t
        phi2 = self.phases[1] + 2 * np.pi * self.freqs[1] * t

        # 4. 合成信号：I = cos1 + cos2, Q = sin1 + sin2
        I = np.cos(phi1) + np.cos(phi2)
        Q = np.sin(phi1) + np.sin(phi2)

        # 5. 更新相位以保证连续性
        self.phases[0] = phi1[-1] % (2 * np.pi)
        self.phases[1] = phi2[-1] % (2 * np.pi)

        # 6. 缩放并转换为 int16
        I16 = (I * self.amp).astype(np.int16)
        Q16 = (Q * self.amp).astype(np.int16)

        # 7. 交错为 [I0, Q0, I1, Q1, ...]
        array_IQ = np.empty(2 * N, dtype=np.int16)
        array_IQ[0::2] = I16
        array_IQ[1::2] = Q16

        return array_IQ.tobytes()


def parse_args():
    parser = argparse.ArgumentParser(description='TCP client sending I/Q sine waves with jitter')
    parser.add_argument('--host', '-H', type=str, default='127.0.0.1',
                        help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', '-p', type=int, default=8000,
                        help='Server port (default: 8000)')
    # 下面的参数可根据需要调整抖动范围和频率
    parser.add_argument('--f1', type=float, default=20,
                        help='Center frequency 1 in Hz (default: 20)')
    parser.add_argument('--f2', type=float, default=30,
                        help='Center frequency 2 in Hz (default: 30)')
    parser.add_argument('--jitter', type=float, default=5.0,
                        help='Max frequency deviation in Hz (default: 5)')
    parser.add_argument('--step', type=float, default=1.0,
                        help='Max freq change per packet in Hz (default: 1)')
    return parser.parse_args()


def pkg_trans(server_host: str, server_port: int, tg: ToneGenerator):
    """
    Connects to server, generates IQ data via ToneGenerator,
    packages [4B len][8B timestamp][2B seq][payload], sends every second.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[{time.time():.4f}] Connected to {server_host}:{server_port}")

    seq_num = 0
    N = 24576 // 2  # number of I/Q pairs per packet

    try:
        while True:
            # get raw IQ bytes
            raw_array = tg.generate(N)

            # header fields
            timestamp = time.time()
            ts_bytes = struct.pack('>d', timestamp)         # 8-byte big-endian double
            seq_bytes = struct.pack('>I', seq_num) # 2-byte big-endian uint16
            length_bytes = len(raw_array).to_bytes(4, 'big')

            # send in one shot
            packet = length_bytes + ts_bytes + seq_bytes + raw_array
            t_start = time.time()
            sock.sendall(packet)
            t_end = time.time()

            print(f"[{t_end:.4f}] Seq={seq_num}, sent {len(packet)} bytes, "
                  f"latency={(t_end - t_start) * 1000:.3f} ms")

            seq_num += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print(f"[{time.time():.4f}] Shutting down pkg_trans")
    finally:
        sock.close()


if __name__ == '__main__':
    args = parse_args()
    # 创建带抖动的正弦波生成器
    tg = ToneGenerator(center_freqs=(args.f1, args.f2),
                       jitter_range=args.jitter,
                       jitter_step=args.step,
                       sample_rate=(24576 // 2))
    pkg_trans(args.host, args.port, tg)
