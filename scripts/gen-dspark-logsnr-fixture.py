#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Generates a tiny synthetic DFlash draft GGUF that carries the four
# DSpark log-SNR conditioning tensors and the matching metadata.
# The weights are zeros, the model is not meant for inference.
# Use it to verify that a loader registers the log-SNR tensors.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))
import gguf

N_EMBD = 32
N_HEAD = 2
N_EMBD_HEAD_K = N_EMBD // N_HEAD
N_FF = 64
N_LAYER = 2
N_CTX = 256
N_VOCAB = 8
N_MARKOV_RANK = 8
TARGET_LAYERS = [0, 1]
N_FREQ = 128
MIN_LOG_SNR = -9.0
MAX_LOG_SNR = 9.0


def add_tensors(writer: gguf.GGUFWriter) -> None:
    n_embd_inp = len(TARGET_LAYERS) * N_EMBD

    specs = {
        'fc.weight': (n_embd_inp, N_EMBD),
        'enc.output_norm.weight': (N_EMBD,),
        'output_norm.weight': (N_EMBD,),
        'markov_w1.weight': (N_MARKOV_RANK, N_VOCAB),
        'markov_w2.weight': (N_MARKOV_RANK, N_VOCAB),
        'log_snr_fc1.weight': (N_FREQ, N_EMBD),
        'log_snr_fc1.bias': (N_EMBD,),
        'log_snr_fc2.weight': (N_EMBD, N_EMBD),
        'log_snr_fc2.bias': (N_EMBD,),
    }

    for i in range(N_LAYER):
        specs.update({
            f'blk.{i}.attn_norm.weight': (N_EMBD,),
            f'blk.{i}.attn_q.weight': (N_EMBD, N_EMBD_HEAD_K * N_HEAD),
            f'blk.{i}.attn_k.weight': (N_EMBD, N_EMBD_HEAD_K * N_HEAD),
            f'blk.{i}.attn_output.weight': (N_EMBD_HEAD_K * N_HEAD, N_EMBD),
            f'blk.{i}.attn_q_norm.weight': (N_EMBD_HEAD_K,),
            f'blk.{i}.attn_k_norm.weight': (N_EMBD_HEAD_K,),
            f'blk.{i}.ffn_norm.weight': (N_EMBD,),
            f'blk.{i}.ffn_gate.weight': (N_EMBD, N_FF),
            f'blk.{i}.ffn_down.weight': (N_FF, N_EMBD),
            f'blk.{i}.ffn_up.weight': (N_EMBD, N_FF),
        })

    for name, ne in specs.items():
        data = np.zeros(ne[::-1], dtype=np.float32) if len(ne) == 2 else np.zeros(ne, dtype=np.float32)
        writer.add_tensor(name, data, raw_shape=ne)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a synthetic DSpark log-SNR draft GGUF fixture')
    parser.add_argument('--out', type=Path, default=Path('dspark-logsnr-fixture.gguf'),
                        help='output GGUF path (default: %(default)s)')
    args = parser.parse_args()

    writer = gguf.GGUFWriter(args.out, arch='dflash')

    writer.add_string('general.name', 'dspark log-snr fixture')
    writer.add_string('tokenizer.ggml.model', 'no_vocab')
    writer.add_uint32('dflash.vocab_size', N_VOCAB)
    writer.add_uint32('dflash.context_length', N_CTX)
    writer.add_uint32('dflash.embedding_length', N_EMBD)
    writer.add_uint32('dflash.block_count', N_LAYER)
    writer.add_uint32('dflash.feed_forward_length', N_FF)
    writer.add_uint32('dflash.attention.head_count', N_HEAD)
    writer.add_float32('dflash.attention.layer_norm_rms_epsilon', 1e-5)
    writer.add_array('dflash.target_layers', TARGET_LAYERS)
    writer.add_bool('dflash.log_snr_conditioning', True)
    writer.add_float32('dflash.min_log_snr', MIN_LOG_SNR)
    writer.add_float32('dflash.max_log_snr', MAX_LOG_SNR)

    add_tensors(writer)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = os.path.getsize(args.out)
    print(f'wrote {args.out} ({size} bytes)')


if __name__ == '__main__':
    main()
