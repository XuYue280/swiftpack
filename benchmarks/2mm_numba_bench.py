import ctypes
import os
import sys
import time
import matplotlib.pyplot as plt
from ctypes import wintypes
from numba import njit, prange
import numpy as np
from threadpoolctl import threadpool_limits

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
# Baseline 1: Naive 2mm in Numba 
# ---------------------------------------------------------
@njit
def _2mm_numba_1(alpha, beta, A, B, C, D):
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
# Baseline 2: Loop Order Permutations
# ---------------------------------------------------------
@njit
def _2mm_2_ijk(alpha, beta, A, B, C, D):
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


@njit
def _2mm_2_ikj(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for i in range(n):
        for j in range(n):
            D[i, j] = beta * D[i, j]
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]


@njit
def _2mm_2_jik(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for j in range(n):
        for i in range(n):
            D[i, j] = beta * D[i, j]
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]


@njit
def _2mm_2_jki(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for j in range(n):
        for i in range(n):
            D[i, j] = beta * D[i, j]
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]


@njit
def _2mm_2_kij(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for i in range(n):
        for j in range(n):
            D[i, j] = beta * D[i, j]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]


@njit
def _2mm_2_kji(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for j in range(n):
        for i in range(n):
            D[i, j] = beta * D[i, j]
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]


# ---------------------------------------------------------
# Baseline 3: Optimization Flags for Backend (IKJ order)
# ---------------------------------------------------------
@njit(fastmath=True)
def _2mm_opt_flags_3(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for i in range(n):
        for j in range(n):
            D[i, j] = beta * D[i, j]
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

# ---------------------------------------------------------
# Baseline 4: Parallel Loop Versions (ikj is the best)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_parallel_i_4(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for i in prange(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for i in prange(n):
        for j in range(n):
            D[i, j] = beta * D[i, j]
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

def _2mm_parallel_k_4(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for k in range(n):
        for i in prange(n):
            for j in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for i in prange(n):
        for j in range(n):
            D[i, j] = beta * D[i, j]
    for k in range(n):
        for i in prange(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

def _2mm_parallel_j_4(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    for j in prange(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += alpha * A[i, k] * B[k, j]

    for j in prange(n):
        for i in range(n):
            D[i, j] = beta * D[i, j]
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

# ---------------------------------------------------------
# Baseline 5: Blocked (Tiled) Parallel I Code
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_blocked_parallel_5(alpha, beta, A, B, C, D):
    n = A.shape[0]
    bs = BLOCK_SIZE
    temp = np.zeros((n, n))

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)
            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)
                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        for j in range(j_block, j_end):
                            temp[i, j] += alpha * A[i, k] * B[k, j]
         
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            for i in range(i_block, i_end):
                for j in range(j_block, j_end):
                    D[i, j] = beta * D[i, j]
        
        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)
            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)
                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        for j in range(j_block, j_end):
                            D[i, j] += temp[i, k] * C[k, j]

# ---------------------------------------------------------
# Baseline 6: Blocked Parallel I using np.dot for Sub-blocks
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_blocked_np_dot_6(alpha, beta, A, B, C, D):
    n = A.shape[0]
    bs = BLOCK_SIZE
    temp = np.zeros((n, n))

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)
            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)
                temp[i_block:i_end, j_block:j_end] += alpha * np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end],
                )

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            D[i_block:i_end, j_block:j_end] *= beta

        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)
            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)
                D[i_block:i_end, j_block:j_end] += np.dot(
                    temp[i_block:i_end, k_block:k_end],
                    C[k_block:k_end, j_block:j_end],
                )
                

# ---------------------------------------------------------
# Baseline 7: Blocked Temp Copy-In / Copy-Out
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_blocked_temp_copy_7(alpha, beta, A, B, C, D):
    n = A.shape[0]
    bs = BLOCK_SIZE
    temp = np.zeros((n, n))

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)

            temp_copy = temp[i_block:i_end, j_block:j_end].copy()
            
            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)
                temp_copy += alpha * np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end],
                )

            temp[i_block:i_end, j_block:j_end] = temp_copy
            
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            d_copy = (beta * D[i_block:i_end, j_block:j_end]).copy()
            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)
                d_copy += np.dot(
                    temp[i_block:i_end, k_block:k_end],
                    C[k_block:k_end, j_block:j_end],
                )
            D[i_block:i_end, j_block:j_end] = d_copy


# ---------------------------------------------------------
# Baseline 8: Two-Level Blocked Parallel I with Temp Copy & np.dot
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_two_level_blocked_temp_np_dot_8(alpha, beta, A, B, C, D):
    n = A.shape[0]
    l2 = BLOCK_SIZE_L2
    l1 = BLOCK_SIZE_L1
    temp = np.zeros((n, n))

    num_l2_i_blocks = (n + l2 - 1) // l2

    for b2 in prange(num_l2_i_blocks):
        i2_start = b2 * l2
        i2_end = min(i2_start + l2, n)

        for j2_start in range(0, n, l2):
            j2_end = min(j2_start + l2, n)

            temp_l2 = temp[i2_start:i2_end, j2_start:j2_end].copy()

            for k2_start in range(0, n, l2):
                k2_end = min(k2_start + l2, n)

                for i1_start in range(i2_start, i2_end, l1):
                    i1_end = min(i1_start + l1, i2_end)
                    i1_rel_start = i1_start - i2_start
                    i1_rel_end = i1_end - i2_start

                    for j1_start in range(j2_start, j2_end, l1):
                        j1_end = min(j1_start + l1, j2_end)
                        j1_rel_start = j1_start - j2_start
                        j1_rel_end = j1_end - j2_start

                        temp_l1 = temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end].copy()

                        for k1_start in range(k2_start, k2_end, l1):
                            k1_end = min(k1_start + l1, k2_end)

                            temp_l1 += alpha * np.dot(
                                A[i1_start:i1_end, k1_start:k1_end],
                                B[k1_start:k1_end, j1_start:j1_end],
                            )

                        temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end] = temp_l1

            temp[i2_start:i2_end, j2_start:j2_end] = temp_l2

        for j2_start in range(0, n, l2):
            j2_end = min(j2_start + l2, n)

            d_l2 = (beta * D[i2_start:i2_end, j2_start:j2_end]).copy()

            for k2_start in range(0, n, l2):
                k2_end = min(k2_start + l2, n)

                for i1_start in range(i2_start, i2_end, l1):
                    i1_end = min(i1_start + l1, i2_end)
                    i1_rel_start = i1_start - i2_start
                    i1_rel_end = i1_end - i2_start

                    for j1_start in range(j2_start, j2_end, l1):
                        j1_end = min(j1_start + l1, j2_end)
                        j1_rel_start = j1_start - j2_start
                        j1_rel_end = j1_end - j2_start

                        d_l1 = d_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end].copy()

                        for k1_start in range(k2_start, k2_end, l1):
                            k1_end = min(k1_start + l1, k2_end)

                            d_l1 += np.dot(
                                temp[i1_start:i1_end, k1_start:k1_end],
                                C[k1_start:k1_end, j1_start:j1_end],
                            )

                        d_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end] = d_l1

            D[i2_start:i2_end, j2_start:j2_end] = d_l2

# ---------------------------------------------------------
# Baseline 9: Zero Allocation Blocked 2mm
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_blocked_zero_alloc_9(alpha, beta, A, B, C, D):
    n = A.shape[0]
    bs = BLOCK_SIZE
    AB = np.zeros((n, n))

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)
        h = i_end - i_block

        temp_tile = np.empty((bs, bs), dtype=A.dtype)
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            w = j_end - j_block

            temp = temp_tile[:h, :w]
            temp.fill(0.0)

            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)

                temp += alpha * np.dot(
                    A[i_block:i_end, k_block:k_end],
                    B[k_block:k_end, j_block:j_end]
                )

            AB[i_block:i_end, j_block:j_end] = temp
        
        for j_block in range(0, n, bs):
            j_end = min(j_block + bs, n)
            w = j_end - j_block
            
            d_copy = (beta * D[i_block:i_end, j_block:j_end]).copy()
            temp = d_copy[:h, :w]

            for k_block in range(0, n, bs):
                k_end = min(k_block + bs, n)
                temp += np.dot(
                    AB[i_block:i_end, k_block:k_end],
                    C[k_block:k_end, j_block:j_end],
                )

            D[i_block:i_end, j_block:j_end] = temp


# ---------------------------------------------------------
# Baseline 10: Reference NumPy dot
# ---------------------------------------------------------
def _2mm_np_dot_10(alpha, beta, A, B, C, D):
    np.copyto(D, alpha * np.dot(np.dot(A, B), C) + beta * D)


# ---------------------------------------------------------
# Baselines 11-15: same np.dot kernel, pinned to P-cores or E-cores.
# One logical processor is used per physical core, so a P-core's
# use AI to discover the below
# ---------------------------------------------------------
class _SYSTEM_CPU_SET_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("Size", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("Id", wintypes.DWORD),
        ("Group", wintypes.WORD),
        ("LogicalProcessorIndex", ctypes.c_ubyte),
        ("CoreIndex", ctypes.c_ubyte),
        ("LastLevelCacheIndex", ctypes.c_ubyte),
        ("NumaNodeIndex", ctypes.c_ubyte),
        ("EfficiencyClass", ctypes.c_ubyte),
        ("AllFlags", ctypes.c_ubyte),
        ("SchedulingClass", wintypes.DWORD),
        ("AllocationTag", ctypes.c_ulonglong),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
_kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
_kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
_kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
_kernel32.GetSystemCpuSetInformation.argtypes = [
    ctypes.c_void_p,
    wintypes.ULONG,
    ctypes.POINTER(wintypes.ULONG),
    wintypes.HANDLE,
    wintypes.ULONG,
]
_kernel32.GetSystemCpuSetInformation.restype = wintypes.BOOL


def _logical_processors_by_efficiency():
    """Return one logical processor id per physical core, grouped by efficiency class."""
    needed = wintypes.ULONG(0)
    _kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), None, 0)
    buf = ctypes.create_string_buffer(needed.value)
    if not _kernel32.GetSystemCpuSetInformation(buf, needed.value, ctypes.byref(needed), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())

    cores = {}
    offset = 0
    raw = buf.raw
    rec_size = ctypes.sizeof(_SYSTEM_CPU_SET_INFORMATION)
    while offset + 8 <= len(raw):
        size = int.from_bytes(raw[offset:offset + 4], "little")
        if size < rec_size:
            break
        rec = _SYSTEM_CPU_SET_INFORMATION.from_buffer_copy(raw[offset:offset + rec_size])
        cores.setdefault(rec.EfficiencyClass, {}).setdefault(rec.CoreIndex, []).append(
            rec.LogicalProcessorIndex
        )
        offset += size

    if len(cores) < 2:
        raise RuntimeError("This CPU does not expose separate P-core and E-core classes")

    def one_thread_per_core(class_id):
        return [min(cores[class_id][core]) for core in sorted(cores[class_id])]

    p_class = max(cores)
    e_class = min(cores)
    return one_thread_per_core(p_class), one_thread_per_core(e_class)


_P_LOGICAL, _E_LOGICAL = _logical_processors_by_efficiency()


def _select_cores(pool, count, kind):
    if len(pool) < count:
        raise RuntimeError(f"Need {count} {kind} cores, found {len(pool)}")
    return pool[:count]


def _2mm_np_dot_on_cores(alpha, beta, A, B, C, D, logical_ids):
    mask = 0
    for lp in logical_ids:
        mask |= 1 << int(lp)

    handle = _kernel32.GetCurrentProcess()
    previous = ctypes.c_size_t()
    system = ctypes.c_size_t()
    if not _kernel32.GetProcessAffinityMask(handle, ctypes.byref(previous), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not _kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with threadpool_limits(limits=len(logical_ids), user_api="blas"):
            np.copyto(D, alpha * np.dot(np.dot(A, B), C) + beta * D)
    finally:
        _kernel32.SetProcessAffinityMask(handle, previous)


def _2mm_6_pcores_11(alpha, beta, A, B, C, D):
    _2mm_np_dot_on_cores(alpha, beta, A, B, C, D, _select_cores(_P_LOGICAL, 6, "P"))


def _2mm_4_pcores_12(alpha, beta, A, B, C, D):
    _2mm_np_dot_on_cores(alpha, beta, A, B, C, D, _select_cores(_P_LOGICAL, 4, "P"))


def _2mm_1_pcore_13(alpha, beta, A, B, C, D):
    _2mm_np_dot_on_cores(alpha, beta, A, B, C, D, _select_cores(_P_LOGICAL, 1, "P"))


def _2mm_4_ecores_14(alpha, beta, A, B, C, D):
    _2mm_np_dot_on_cores(alpha, beta, A, B, C, D, _select_cores(_E_LOGICAL, 4, "E"))


def _2mm_1_ecore_15(alpha, beta, A, B, C, D):
    _2mm_np_dot_on_cores(alpha, beta, A, B, C, D, _select_cores(_E_LOGICAL, 1, "E"))


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

    D_org = D.copy()

    # Compute ground truth reference solution for correctness verification
    D_expected = alpha * np.dot(np.dot(A, B), C) + beta * D
    total_flops = 5.0 * (N ** 3) + (N ** 2) # 3N^3 (for alpha * A * B), N^2 (for beta * D), 2N^3 (for temp * C)

    def measure(fn, warmup=True, reps=9):
        if warmup:
            D[:] = D_org
            fn(alpha, beta, A, B, C, D)

        # Correctness check against reference matrix C_expected
        D[:] = D_org
        fn(alpha, beta, A, B, C, D)
        is_correct = np.allclose(D, D_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            D[:] = D_org
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
        ("1_numba_naive", _2mm_numba_1),
        ("2_order_ijk", _2mm_2_ijk),
        ("2_order_ikj", _2mm_2_ikj),
        ("2_order_jik", _2mm_2_jik),
        ("2_order_jki", _2mm_2_jki),
        ("2_order_kij", _2mm_2_kij),
        ("2_order_kji", _2mm_2_kji),
        ("3_fastmath_ikj", _2mm_opt_flags_3),
        ("4_parallel_i", _2mm_parallel_i_4),
        #("4_parallel_k", _2mm_parallel_k_4),
        #("4_parallel_j", _2mm_parallel_j_4),
        ("5_blocked_parallel_i", _2mm_blocked_parallel_5),
        ("6_blocked_np_dot", _2mm_blocked_np_dot_6),
        ("7_blocked_temp_copy", _2mm_blocked_temp_copy_7),
        ("8_two_level_blocked_temp_np_dot", _2mm_two_level_blocked_temp_np_dot_8),
        ("9_blocked_zero_alloc", _2mm_blocked_zero_alloc_9),
        ("10_np_dot", _2mm_np_dot_10),
        ("11_6_pcores", _2mm_6_pcores_11),
        ("12_4_pcores", _2mm_4_pcores_12),
        ("13_1_pcore", _2mm_1_pcore_13),
        ("14_4_ecores", _2mm_4_ecores_14),
        ("15_1_ecore", _2mm_1_ecore_15),
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