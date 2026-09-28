import os
import sys
import time
import matplotlib.pyplot as plt
from numba import njit, prange
import numpy as np

# Set single-threaded execution for external BLAS libraries to ensure reproducible benchmarks
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Default Matrix Dimensions & Block Sizes
N = 512
BLOCK_SIZE = 32
BLOCK_SIZE_L2 = 128
BLOCK_SIZE_L1 = 32

# 2MM is defined as
# D := alpha * A * B * C + beta * D, where alpha and beta are scalars, and A, B, C, D are matrices.
# 1. temp = alpha * A * B
# 2. D := temp * C + beta * D


# ---------------------------------------------------------
# Baseline 0: Pure Python Naive 2mm 
# ---------------------------------------------------------
def _2mm_python_0(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j] 

    for i in range(n):
        for j in range(n):
            D[i, j] = beta * D[i, j]
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]














# ---------------------------------------------------------
# Execution & Benchmarking Routine
# ---------------------------------------------------------
def run_benchmark(matrix_size=512):
    global N
    N = matrix_size

    np.random.seed(42)
    alpha = np.float64(np.random.uniform(-1.0, 1.0))
    beta = np.float64(np.random.uniform(-1.0, 1.0))
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.random.randn(N, N).astype(np.float64)
    D = np.random.randn(N, N).astype(np.float64)

    # Compute ground truth reference solution for correctness verification
    D_expected = alpha * np.dot(np.dot(A, B), C) + beta * D
    total_flops = 5.0 * (N ** 3) + (N ** 2) # 3N^3 (for alpha * A * B), N^2 (for beta * D), 2N^3 (for temp * C)

    def measure(fn, warmup=True, reps=9):
        if warmup:
            fn(alpha, beta, A, B, C, D)

        # Correctness check against reference matrix C_expected
        fn(alpha, beta, A, B, C, D)
        is_correct = np.allclose(D, D_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            fn(alpha, beta, A, B, C, D)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python (Run on a smaller dimension if N is large to prevent lockups)
    N_py = min(N, 128)
    A_py, B_py, C_py, D_py = A[:N_py, :N_py].copy(), B[:N_py, :N_py].copy(), C[:N_py, :N_py].copy(), D[:N_py, :N_py].copy()
    D_py_expected = alpha * np.dot(np.dot(A_py, B_py), C_py) + beta * D_py

    t0 = time.perf_counter()
    _2mm_python_0(alpha, beta, A_py, B_py, C_py, D_py)
    py_elapsed = time.perf_counter() - t0
    py_gflops = (total_flops / py_elapsed) / 1e9
    py_correct = np.allclose(D_py, D_py_expected, rtol=1e-5, atol=1e-5)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        # ("1_numba_naive", xxx),
        # ("2_order_ijk", matmul_2_ijk),
        # ("2_order_ikj", matmul_2_ikj),
        # ("2_order_jik", matmul_2_jik),
        # ("2_order_jki", matmul_2_jki),
        # ("2_order_kij", matmul_2_kij),
        # ("2_order_kji", matmul_2_kji),
        # ("3_fastmath_ikj", matmul_opt_flags_3),
        # ("4_parallel_i", matmul_parallel_i_4),
        # ("4_parallel_k", matmul_parallel_k_4),
        # ("4_parallel_j", matmul_parallel_j_4),
        # ("5_blocked_parallel_i", matmul_blocked_parallel_5),
        # ("6_blocked_np_dot", matmul_blocked_np_dot_7),
        # ("7_blocked_temp_copy", matmul_blocked_temp_copy_8),
        # ("8_two_level_blocked_temp_np_dot", matmul_two_level_blocked_temp_np_dot_9),
        # ("9_blocked_zero_alloc", matmul_blocked_zero_alloc_8),
        # ("10_np_dot", matmul_np_dot),
    ]

    for name, fn in functions:
        gflops, elapsed, is_correct = measure(fn)
        results.append((name, gflops, elapsed, is_correct))

    return results

# ---------------------------------------------------------
# Formatting and Output Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    # Get user input for matrix dimension N
    if len(sys.argv) > 1:
        try:
            N_input = int(sys.argv[1])
        except ValueError:
            N_input = 512
    else:
        N_input = 512

    print(f"\nRunning Matrix Multiplication Benchmarks for Matrix Size {N_input}x{N_input}...\n")
    benchmark_data = run_benchmark(N_input)

    py_elapsed = benchmark_data[0][2]  # Reference execution time for pure Python naive

    # Table Header Formatting
    header = f"| {'Baseline Implementation':<33} | {'GFLOP/s':<10} | {'Abs Speedup':<12} | {'Rel Speedup':<12} | {'Correct':<8} |"
    divider = "-" * len(header)

    print(divider)
    print(header)
    print(divider)

    prev_elapsed = None

    for name, gflops, elapsed, is_correct in benchmark_data:
        abs_speedup = py_elapsed / elapsed
        rel_speedup = (prev_elapsed / elapsed) if prev_elapsed is not None else 1.0
        status = "PASS" if is_correct else "FAIL"

        print(f"| {name:<33} | {gflops:10.3f} | {abs_speedup:12.2f}x | {rel_speedup:12.2f}x | {status:<8} |")
        prev_elapsed = elapsed

    print(divider)

    # Plotting Output
    names = [row[0] for row in benchmark_data]
    gflops_vals = [row[1] for row in benchmark_data]

    plt.figure(figsize=(16, 6))
    bars = plt.bar(names, gflops_vals, color="skyblue", edgecolor="navy")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (0.02 * max(gflops_vals)),
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.ylabel("GFLOP/s (Higher is better)")
    plt.title(f"Numba Matrix Multiplication Benchmark Performance (N={N_input})")
    plt.xticks(rotation=45, ha="right")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()