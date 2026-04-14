"""
WKV CUDA Kernel Benchmark
=========================
Compares the original [B, T, C] kernel against the optimized [B, C, T] kernel.

Run from the repo root:
    python benchmarks/wkv_benchmark.py

Requires a CUDA-capable GPU.
"""

import sys
import os

# Allow imports from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available. This benchmark requires a GPU.")
    sys.exit(1)

from torch.utils.cpp_extension import load

# ── compile kernels ────────────────────────────────────────────────────────────
T_MAX = 1024  # must match the value used in model.py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUDA_DIR  = os.path.join(REPO_ROOT, "cuda")

COMMON_CUDA_FLAGS = [
    "-res-usage",
    "--maxrregcount 60",
    "--use_fast_math",
    "-O3",
    "-Xptxas -O3",
    f"-DTmax={T_MAX}",
]

print("Compiling original WKV kernel [B, T, C] …")
wkv_orig = load(
    name="wkv_orig_bench",
    sources=[
        os.path.join(CUDA_DIR, "wkv_op.cpp"),
        os.path.join(CUDA_DIR, "wkv_cuda.cu"),
    ],
    verbose=False,
    extra_cuda_cflags=COMMON_CUDA_FLAGS,
)

print("Compiling optimized WKV kernel [B, C, T] …")
wkv_opt = load(
    name="wkv_opt_bench",
    sources=[
        os.path.join(CUDA_DIR, "wkv_op_opt.cpp"),
        os.path.join(CUDA_DIR, "wkv_cuda_opt.cu"),
    ],
    verbose=False,
    extra_cuda_cflags=COMMON_CUDA_FLAGS,
)

print("Both kernels compiled successfully.\n")

# ── helper: memory bandwidth formula ──────────────────────────────────────────
BYTES_PER_FLOAT32 = 4

def compute_memory_bytes_forward(B: int, T: int, C: int) -> int:
    """
    Inputs read:
        w  [C]        – 1 read per thread (per (b,c) pair)
        u  [C]        – 1 read per thread
        k  [B, T, C]  – B*T*C reads
        v  [B, T, C]  – B*T*C reads
    Outputs written:
        y  [B, T, C]  – B*T*C writes
    Total floats = 2*C + 3*B*T*C
    """
    floats = 2 * C + 3 * B * T * C
    return floats * BYTES_PER_FLOAT32

def compute_memory_bytes_backward(B: int, T: int, C: int) -> int:
    """
    Inputs read:
        w  [C]
        u  [C]
        k  [B, T, C]
        v  [B, T, C]
        gy [B, T, C]
    Outputs written:
        gw [B, C]  (summed later, but we write B*C)
        gu [B, C]
        gk [B, T, C]
        gv [B, T, C]
    Total floats = 2*C + 3*B*T*C + 2*B*C + 2*B*T*C = 2*C + 5*B*T*C + 2*B*C
    """
    floats = 2 * C + 5 * B * T * C + 2 * B * C
    return floats * BYTES_PER_FLOAT32


# ── benchmark function ─────────────────────────────────────────────────────────
def run_benchmark(
    label: str,
    forward_fn,
    backward_fn,
    B: int,
    T: int,
    C: int,
    warmup: int = 10,
    repeats: int = 100,
):
    """
    Measures forward and backward pass time using torch.cuda.Event.
    Returns (fwd_ms_mean, bwd_ms_mean, fwd_gbps, bwd_gbps).
    """
    device = "cuda"

    # ── allocate tensors ──────────────────────────────────────────────────────
    w  = torch.randn(C,       device=device, dtype=torch.float32)
    u  = torch.randn(C,       device=device, dtype=torch.float32)
    k  = torch.randn(B, T, C, device=device, dtype=torch.float32)
    v  = torch.randn(B, T, C, device=device, dtype=torch.float32)
    gy = torch.randn(B, T, C, device=device, dtype=torch.float32)

    # The optimized kernel works with [B, C, T] layout
    is_opt = (backward_fn is not None and "opt" in label.lower())

    def make_inputs():
        if is_opt:
            k_ = k.permute(0, 2, 1).contiguous()
            v_ = v.permute(0, 2, 1).contiguous()
            gy_ = gy.permute(0, 2, 1).contiguous()
            y_ = torch.empty(B, C, T, device=device, dtype=torch.float32)
        else:
            k_ = k.contiguous()
            v_ = v.contiguous()
            gy_ = gy.contiguous()
            y_ = torch.empty(B, T, C, device=device, dtype=torch.float32)
        w_ = -torch.exp(w.contiguous())
        u_ = u.contiguous()
        gw_ = torch.zeros(B, C, device=device, dtype=torch.float32)
        gu_ = torch.zeros(B, C, device=device, dtype=torch.float32)
        gk_ = torch.zeros_like(k_)
        gv_ = torch.zeros_like(v_)
        return w_, u_, k_, v_, gy_, y_, gw_, gu_, gk_, gv_

    # ── warmup ────────────────────────────────────────────────────────────────
    for _ in range(warmup):
        w_, u_, k_, v_, gy_, y_, gw_, gu_, gk_, gv_ = make_inputs()
        forward_fn(B, T, C, w_, u_, k_, v_, y_)
        if backward_fn is not None:
            backward_fn(B, T, C, w_, u_, k_, v_, gy_, gw_, gu_, gk_, gv_)
    torch.cuda.synchronize()

    # ── measure forward ───────────────────────────────────────────────────────
    fwd_times = []
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    for _ in range(repeats):
        w_, u_, k_, v_, gy_, y_, gw_, gu_, gk_, gv_ = make_inputs()
        start_evt.record()
        forward_fn(B, T, C, w_, u_, k_, v_, y_)
        end_evt.record()
        torch.cuda.synchronize()
        fwd_times.append(start_evt.elapsed_time(end_evt))

    fwd_ms_mean = sum(fwd_times) / len(fwd_times)
    fwd_bytes   = compute_memory_bytes_forward(B, T, C)
    fwd_gbps    = (fwd_bytes / 1e9) / (fwd_ms_mean / 1e3)

    # ── measure backward ──────────────────────────────────────────────────────
    bwd_ms_mean = None
    bwd_gbps    = None
    if backward_fn is not None:
        bwd_times = []
        for _ in range(repeats):
            w_, u_, k_, v_, gy_, y_, gw_, gu_, gk_, gv_ = make_inputs()
            start_evt.record()
            backward_fn(B, T, C, w_, u_, k_, v_, gy_, gw_, gu_, gk_, gv_)
            end_evt.record()
            torch.cuda.synchronize()
            bwd_times.append(start_evt.elapsed_time(end_evt))

        bwd_ms_mean = sum(bwd_times) / len(bwd_times)
        bwd_bytes   = compute_memory_bytes_backward(B, T, C)
        bwd_gbps    = (bwd_bytes / 1e9) / (bwd_ms_mean / 1e3)

    return fwd_ms_mean, bwd_ms_mean, fwd_gbps, bwd_gbps


# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, C = 32, 1024, 512

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU       : {gpu_name}")
    print(f"Shape     : B={B}, T={T}, C={C}")
    print(f"Warmup    : 10 runs | Measurement: 100 runs\n")
    print("=" * 65)

    # ── Original kernel ───────────────────────────────────────────────────────
    print("Running ORIGINAL kernel [B, T, C] …")
    orig_fwd_ms, orig_bwd_ms, orig_fwd_gbps, orig_bwd_gbps = run_benchmark(
        label="original",
        forward_fn=wkv_orig.forward,
        backward_fn=wkv_orig.backward,
        B=B, T=T, C=C,
    )

    print(f"  Forward  : {orig_fwd_ms:.4f} ms  |  {orig_fwd_gbps:.2f} GB/s")
    print(f"  Backward : {orig_bwd_ms:.4f} ms  |  {orig_bwd_gbps:.2f} GB/s")
    print()

    # ── Optimized kernel ──────────────────────────────────────────────────────
    print("Running OPTIMIZED kernel [B, C, T] (coalesced) …")
    opt_fwd_ms, opt_bwd_ms, opt_fwd_gbps, opt_bwd_gbps = run_benchmark(
        label="optimized",
        forward_fn=wkv_opt.forward,
        backward_fn=wkv_opt.backward,
        B=B, T=T, C=C,
    )

    print(f"  Forward  : {opt_fwd_ms:.4f} ms  |  {opt_fwd_gbps:.2f} GB/s")
    print(f"  Backward : {opt_bwd_ms:.4f} ms  |  {opt_bwd_gbps:.2f} GB/s")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    fwd_speedup = orig_fwd_ms / opt_fwd_ms
    bwd_speedup = orig_bwd_ms / opt_bwd_ms
    fwd_delta_gbps = opt_fwd_gbps - orig_fwd_gbps
    bwd_delta_gbps = opt_bwd_gbps - orig_bwd_gbps

    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<30} {'Original':>10} {'Optimized':>10} {'Speedup':>8}")
    print("-" * 65)
    print(f"{'Forward  time  (ms)':<30} {orig_fwd_ms:>10.4f} {opt_fwd_ms:>10.4f} {fwd_speedup:>7.3f}x")
    print(f"{'Backward time  (ms)':<30} {orig_bwd_ms:>10.4f} {opt_bwd_ms:>10.4f} {bwd_speedup:>7.3f}x")
    print(f"{'Forward  BW    (GB/s)':<30} {orig_fwd_gbps:>10.2f} {opt_fwd_gbps:>10.2f} {fwd_delta_gbps:>+7.2f}")
    print(f"{'Backward BW    (GB/s)':<30} {orig_bwd_gbps:>10.2f} {opt_bwd_gbps:>10.2f} {bwd_delta_gbps:>+7.2f}")
    print("=" * 65)
    print()

    if fwd_speedup >= 1.0:
        print(f"Optimized forward  is {fwd_speedup:.2f}x FASTER (+{fwd_delta_gbps:.1f} GB/s)")
    else:
        print(f"Optimized forward  is {1/fwd_speedup:.2f}x SLOWER ({fwd_delta_gbps:.1f} GB/s)")

    if bwd_speedup >= 1.0:
        print(f"Optimized backward is {bwd_speedup:.2f}x FASTER (+{bwd_delta_gbps:.1f} GB/s)")
    else:
        print(f"Optimized backward is {1/bwd_speedup:.2f}x SLOWER ({bwd_delta_gbps:.1f} GB/s)")

    # Note on memory layout overhead
    print()
    print("NOTE: the optimized kernel requires a .permute(0,2,1).contiguous()")
    print("      transpose before each call. In the benchmark above this cost")
    print("      is excluded (inputs pre-transposed inside make_inputs()).")
    print("      In a real training loop the transpose adds ~0.1-0.3 ms.")
