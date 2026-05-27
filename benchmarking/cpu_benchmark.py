"""
CPU Benchmark for bitsandbytes on Windows ARM64.

Tests quantization/dequantization performance for the CPU backend.

Usage: python benchmarking/cpu_benchmark.py
"""

import time
import statistics

import numpy as np
import torch

import bitsandbytes as bnb
from bitsandbytes import functional as F


def benchmark_fn(fn, *args, warmup=3, iterations=10, **kwargs):
    """Run a function multiple times and return timing statistics."""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_ms": statistics.mean(times) * 1000,
        "std_ms": statistics.stdev(times) * 1000 if len(times) > 1 else 0,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "iterations": iterations,
    }


def format_result(name, result):
    return (
        f"{name:50s} | "
        f"mean: {result['mean_ms']:8.2f} ms | "
        f"std: {result['std_ms']:6.2f} ms | "
        f"min: {result['min_ms']:8.2f} ms | "
        f"max: {result['max_ms']:8.2f} ms"
    )


def run_benchmarks():
    results = []
    print("=" * 100)
    print("bitsandbytes CPU Benchmark - Windows ARM64")
    print("=" * 100)
    print(f"PyTorch version: {torch.__version__}")
    print(f"bitsandbytes version: {bnb.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"Device: CPU")
    print(f"Torch threads: {torch.get_num_threads()}")
    print("=" * 100)
    print()

    # =========================================================================
    # 8-bit Quantization Benchmarks
    # =========================================================================
    print("-" * 100)
    print("8-bit Quantization (blockwise)")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.float32, device="cpu")
        code = F.create_dynamic_map().to("cpu")

        name = f"quantize_blockwise_fp32 [{rows}x{cols}]"
        result = benchmark_fn(F.quantize_blockwise, A, code=code)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # 8-bit Dequantization Benchmarks
    # =========================================================================
    print("-" * 100)
    print("8-bit Dequantization (blockwise)")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.float32, device="cpu")
        code = F.create_dynamic_map().to("cpu")
        quantized, state = F.quantize_blockwise(A, code=code)

        name = f"dequantize_blockwise_fp32 [{rows}x{cols}]"
        result = benchmark_fn(F.dequantize_blockwise, quantized, quant_state=state)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # 4-bit NF4 Quantization Benchmarks
    # =========================================================================
    print("-" * 100)
    print("4-bit NF4 Quantization")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.float32, device="cpu")

        name = f"quantize_nf4_fp32 [{rows}x{cols}]"
        result = benchmark_fn(F.quantize_nf4, A)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # 4-bit NF4 Dequantization Benchmarks
    # =========================================================================
    print("-" * 100)
    print("4-bit NF4 Dequantization")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.float32, device="cpu")
        quantized, state = F.quantize_nf4(A)

        name = f"dequantize_nf4_fp32 [{rows}x{cols}]"
        result = benchmark_fn(F.dequantize_nf4, quantized, quant_state=state)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # 4-bit FP4 Quantization Benchmarks
    # =========================================================================
    print("-" * 100)
    print("4-bit FP4 Quantization")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.float32, device="cpu")

        name = f"quantize_fp4_fp32 [{rows}x{cols}]"
        result = benchmark_fn(F.quantize_fp4, A)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # 4-bit FP4 Dequantization Benchmarks
    # =========================================================================
    print("-" * 100)
    print("4-bit FP4 Dequantization")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.float32, device="cpu")
        quantized, state = F.quantize_fp4(A)

        name = f"dequantize_fp4_fp32 [{rows}x{cols}]"
        result = benchmark_fn(F.dequantize_fp4, quantized, quant_state=state)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # Half-precision (float16/bfloat16) 4-bit Dequantization
    # =========================================================================
    print("-" * 100)
    print("4-bit NF4 Dequantization (bfloat16 output)")
    print("-" * 100)

    for size in [(1024, 1024), (4096, 4096), (1024, 8192)]:
        rows, cols = size
        A = torch.randn(rows, cols, dtype=torch.bfloat16, device="cpu")
        quantized, state = F.quantize_nf4(A)

        name = f"dequantize_nf4_bf16 [{rows}x{cols}]"
        result = benchmark_fn(F.dequantize_nf4, quantized, quant_state=state)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # Linear4bit forward pass benchmark
    # =========================================================================
    print("-" * 100)
    print("Linear4bit Forward Pass (NF4)")
    print("-" * 100)

    for in_features, out_features, batch_size in [
        (1024, 1024, 1),
        (4096, 4096, 1),
        (4096, 11008, 1),
        (4096, 4096, 8),
        (4096, 11008, 8),
    ]:
        linear = bnb.nn.Linear4bit(
            in_features, out_features, bias=False,
            compute_dtype=torch.float32,
            quant_type="nf4",
        )
        # Quantize the weights
        linear = linear.to("cpu")
        # Force quantization by doing a forward pass
        x = torch.randn(batch_size, in_features, dtype=torch.float32, device="cpu")
        _ = linear(x)  # trigger quantization

        name = f"Linear4bit_nf4_fwd [bs={batch_size}, {in_features}->{out_features}]"
        result = benchmark_fn(linear, x, warmup=2, iterations=5)
        results.append((name, result))
        print(format_result(name, result))

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for name, result in results:
        print(format_result(name, result))
    print("=" * 100)

    return results


if __name__ == "__main__":
    run_benchmarks()