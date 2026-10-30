

import ctypes
import os
import sys
import time
import matplotlib.pyplot as plt
from ctypes import wintypes
from numba import njit, prange
import numpy as np
from threadpoolctl import threadpool_limits

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# Default Matrix Dimensions & Block Sizes
N = 512
BLOCK_SIZE = 64
BLOCK_SIZE_L2 = 128
BLOCK_SIZE_L1 = 32

# 2MM is defined as
# D := alpha * A * B * C + beta * D, where alpha and beta are scalars, and A, B, C, D are matrices.
# 1. A' = alpha * A
# 2. temp = A' * B
# 3. D := temp * C + beta * D

# total_flops = 4N^3 + 2N^2


# ---------------------------------------------------------
# Baseline 0: Pure Python Naive 2mm 
# ---------------------------------------------------------
def _2mm_python_0(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))
    
    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]


# ---------------------------------------------------------
# Baseline 1: Naive 2mm in Numba 
# ---------------------------------------------------------
@njit
def _2mm_numba_1(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]



# ---------------------------------------------------------
# Baseline 2: Loop Order Permutations (6 x 6 = 36 combinations)
# _2mm_<first>_<second>_2: <first> is the loop order of temp = A' * B,
# <second> is the loop order of D = temp * C + beta * D.
# alpha / beta scaling is a separate N^2 loop so only the loop order differs.
# ---------------------------------------------------------
@njit
def _2mm_ijk_ijk_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]


@njit
def _2mm_ijk_ikj_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ijk_jik_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ijk_jki_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ijk_kij_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ijk_kji_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ikj_ijk_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ikj_ikj_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ikj_jik_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ikj_jki_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ikj_kij_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_ikj_kji_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jik_ijk_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jik_ikj_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jik_jik_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jik_jki_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jik_kij_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jik_kji_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jki_ijk_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jki_ikj_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jki_jik_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jki_jki_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jki_kij_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_jki_kji_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kij_ijk_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kij_ikj_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kij_jik_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kij_jki_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kij_kij_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kij_kji_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kji_ijk_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ijk
    for i in range(n):
        for j in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kji_ikj_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order ikj
    for i in range(n):
        for k in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kji_jik_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jik
    for j in range(n):
        for i in range(n):
            for k in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kji_jki_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order jki
    for j in range(n):
        for k in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kji_kij_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kij
    for k in range(n):
        for i in range(n):
            for j in range(n):
                D[i, j] += temp[i, k] * C[k, j]

@njit
def _2mm_kji_kji_2(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n))

    # 2N^2
    for i in range(n):
        for j in range(n):
            A[i, j] *= alpha
            D[i, j] *= beta

    # 2N^3: temp = A' * B, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                temp[i, j] += A[i, k] * B[k, j]

    # 2N^3: D += temp * C, loop order kji
    for k in range(n):
        for j in range(n):
            for i in range(n):
                D[i, j] += temp[i, k] * C[k, j]



# ---------------------------------------------------------
# Baseline 3: loop fusion, ikj
# ---------------------------------------------------------
@njit
def _2mm_fusion_ikj_3(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp_row = np.zeros(n, dtype=A.dtype)

    for i in range(n):
        temp_row[:] = 0

        for k in range(n): #2N^2
            A[i, k] *= alpha
            D[i, k] *= beta
            a = A[i, k]

            for j in range(n): #2N^3
                temp_row[j] += a * B[k, j]


        for k in range(n):
            t = temp_row[k]

            for j in range(n): #2N^3
                D[i, j] += t * C[k, j]



# ---------------------------------------------------------
# Baseline 4: Optimization Flags for ikj fusion
# ---------------------------------------------------------
@njit(fastmath=True)
def _2mm_opt_flags_fusion_ikj_4(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp_row = np.zeros(n, dtype=A.dtype)

    for i in range(n):
        temp_row[:] = 0

        for k in range(n): #2N^2
            A[i, k] *= alpha
            D[i, k] *= beta
            a = A[i, k]

            for j in range(n): #2N^3
                temp_row[j] += a * B[k, j]


        for k in range(n):
            t = temp_row[k]

            for j in range(n): #2N^3
                D[i, j] += t * C[k, j]


# ---------------------------------------------------------
# Baseline 5: Parallel Loop Versions (ikj is the best)
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_parallel_i_5(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n), dtype=A.dtype)

    for i in prange(n):
        for k in prange(n):
            A[i, k] *= alpha
            a = A[i, k]
            for j in prange(n):
                temp[i, j] += a * B[k, j]

    for i in prange(n):
        for j in prange(n):
            D[i, j] *= beta
        for k in prange(n):
            t = temp[i, k]
            for j in prange(n):
                D[i, j] += t * C[k, j]


@njit(parallel=True, fastmath=True)
def _2mm_parallel_fusion_ikj_5(alpha, beta, A, B, C, D):
    n = A.shape[0]
    temp = np.zeros((n, n), dtype=A.dtype)

    for i in prange(n):
        for k in prange(n): #2N^2
            A[i, k] *= alpha
            D[i, k] *= beta
            a = A[i, k]

            for j in prange(n): #2N^3
                temp[i, j] += a * B[k, j]
            

        for k in prange(n):
            t = temp[i, k]

            for j in prange(n): #2N^3
                D[i, j] += t * C[k, j]


# ---------------------------------------------------------
# Baseline 6: Blocked (Tiled) Parallel I Code
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_blocked_parallel_6(alpha, beta, A, B, C, D):
    n = A.shape[0]
    bs = BLOCK_SIZE
    temp = np.zeros((n, n), dtype=A.dtype)

    num_i_blocks = (n + bs - 1) // bs

    # temp = (alpha * A) @ B
    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for i in range(i_block, i_end):
            for k in range(n):
                A[i, k] *= alpha

        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)

            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)

                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        a = A[i, k]
                        for j in range(j_block, j_end):
                            temp[i, j] += a * B[k, j]

    # D = beta * D + temp @ C
    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        for i in range(i_block, i_end):
            for j in range(n):
                D[i, j] *= beta

        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)

            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)

                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        t = temp[i, k]
                        for j in range(j_block, j_end):
                            D[i, j] += t * C[k, j]




@njit(parallel=True, fastmath=True)
def _2mm_parallel_fusion_blocked_6(alpha, beta, A, B, C, D):
    n = A.shape[0]
    bs = BLOCK_SIZE
    temp = np.zeros((n, n), dtype=A.dtype)

    num_i_blocks = (n + bs - 1) // bs

    for b in prange(num_i_blocks):
        i_block = b * bs
        i_end = min(i_block + bs, n)

        # scale up
        for i in range(i_block, i_end):
            for j in range(n):
                A[i, j] *= alpha
                D[i, j] *= beta

        # temp = A @ B
        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)

            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)

                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        a = A[i, k]
                        for j in range(j_block, j_end):
                            temp[i, j] += a * B[k, j]

        # D += temp @ C
        for k_block in range(0, n, bs):
            k_end = min(k_block + bs, n)

            for j_block in range(0, n, bs):
                j_end = min(j_block + bs, n)

                for i in range(i_block, i_end):
                    for k in range(k_block, k_end):
                        t = temp[i, k]
                        for j in range(j_block, j_end):
                            D[i, j] += t * C[k, j]


# ---------------------------------------------------------
# Baseline 8: loop unrolling
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_loop_unrolling_8(alpha, beta, A, B, C, D):
    n = A.shape[0]
    n4 = n - n % 4

    for i in prange(n):
        temp_row = np.zeros(n, dtype=A.dtype)

        for k in range(0, n4, 4):
            A[i, k]     *= alpha
            A[i, k + 1] *= alpha
            A[i, k + 2] *= alpha
            A[i, k + 3] *= alpha

            D[i, k]     *= beta
            D[i, k + 1] *= beta
            D[i, k + 2] *= beta
            D[i, k + 3] *= beta

            a0 = A[i, k]
            a1 = A[i, k + 1]
            a2 = A[i, k + 2]
            a3 = A[i, k + 3]

            for j in prange(n):
                temp_row[j] += a0 * B[k, j]
                temp_row[j] += a1 * B[k + 1, j]
                temp_row[j] += a2 * B[k + 2, j]
                temp_row[j] += a3 * B[k + 3, j]

        # tail
        for k in range(n4, n):
            A[i, k] *= alpha
            D[i, k] *= beta
            a = A[i, k]

            for j in prange(n):
                temp_row[j] += a * B[k, j]



        for k in range(0, n4, 4):
            t0 = temp_row[k]
            t1 = temp_row[k + 1]
            t2 = temp_row[k + 2]
            t3 = temp_row[k + 3]

            for j in prange(n):
                D[i, j] += t0 * C[k, j]
                D[i, j] += t1 * C[k + 1, j]
                D[i, j] += t2 * C[k + 2, j]
                D[i, j] += t3 * C[k + 3, j]

        # tail
        for k in range(n4, n):
            t = temp_row[k]

            for j in prange(n):
                D[i, j] += t * C[k, j]


# ---------------------------------------------------------
# Baseline 9: micro kernel 2x2
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_micro_kernel_2x2_9(alpha, beta, A, B, C, D):
    n = A.shape[0]

    for row_pair in prange(n // 2):
        i = row_pair * 2
        temp = np.zeros((2, n), dtype=A.dtype)

        # scale up
        for j in prange(n):
            A[i, j]     *= alpha
            A[i + 1, j] *= alpha

            D[i, j]     *= beta
            D[i + 1, j] *= beta

        # temp = A @ B
        for k in prange(n):
            a0 = A[i, k]
            a1 = A[i + 1, k]

            for j in range(0, n, 2):
                b0 = B[k, j]
                b1 = B[k, j + 1]

                temp[0, j]     += a0 * b0
                temp[0, j + 1] += a0 * b1
                temp[1, j]     += a1 * b0
                temp[1, j + 1] += a1 * b1

        # D += temp @ C
        for k in prange(n):
            t0 = temp[0, k]
            t1 = temp[1, k]

            for j in range(0, n, 2):
                c0 = C[k, j]
                c1 = C[k, j + 1]

                D[i, j]         += t0 * c0
                D[i, j + 1]     += t0 * c1
                D[i + 1, j]     += t1 * c0
                D[i + 1, j + 1] += t1 * c1

@njit(parallel=True, fastmath=True)
def _2mm_micro_kernel_4x4_9(alpha, beta, A, B, C, D):
    n = A.shape[0]

    for row_group in prange(n // 4):
        i = row_group * 4
        temp = np.zeros((4, n), dtype=A.dtype)

        # scale up
        for j in prange(n):
            A[i, j]     *= alpha
            A[i + 1, j] *= alpha
            A[i + 2, j] *= alpha
            A[i + 3, j] *= alpha

            D[i, j]     *= beta
            D[i + 1, j] *= beta
            D[i + 2, j] *= beta
            D[i + 3, j] *= beta

        # temp = A @ B
        for k in prange(n):
            a0 = A[i, k]
            a1 = A[i + 1, k]
            a2 = A[i + 2, k]
            a3 = A[i + 3, k]

            for j in range(0, n, 4):
                b0 = B[k, j]
                b1 = B[k, j + 1]
                b2 = B[k, j + 2]
                b3 = B[k, j + 3]

                temp[0, j]     += a0 * b0
                temp[0, j + 1] += a0 * b1
                temp[0, j + 2] += a0 * b2
                temp[0, j + 3] += a0 * b3

                temp[1, j]     += a1 * b0
                temp[1, j + 1] += a1 * b1
                temp[1, j + 2] += a1 * b2
                temp[1, j + 3] += a1 * b3

                temp[2, j]     += a2 * b0
                temp[2, j + 1] += a2 * b1
                temp[2, j + 2] += a2 * b2
                temp[2, j + 3] += a2 * b3

                temp[3, j]     += a3 * b0
                temp[3, j + 1] += a3 * b1
                temp[3, j + 2] += a3 * b2
                temp[3, j + 3] += a3 * b3

        # D += temp @ C
        for k in prange(n):
            t0 = temp[0, k]
            t1 = temp[1, k]
            t2 = temp[2, k]
            t3 = temp[3, k]

            for j in range(0, n, 4):
                c0 = C[k, j]
                c1 = C[k, j + 1]
                c2 = C[k, j + 2]
                c3 = C[k, j + 3]

                D[i, j]     += t0 * c0
                D[i, j + 1] += t0 * c1
                D[i, j + 2] += t0 * c2
                D[i, j + 3] += t0 * c3

                D[i + 1, j]     += t1 * c0
                D[i + 1, j + 1] += t1 * c1
                D[i + 1, j + 2] += t1 * c2
                D[i + 1, j + 3] += t1 * c3

                D[i + 2, j]     += t2 * c0
                D[i + 2, j + 1] += t2 * c1
                D[i + 2, j + 2] += t2 * c2
                D[i + 2, j + 3] += t2 * c3

                D[i + 3, j]     += t3 * c0
                D[i + 3, j + 1] += t3 * c1
                D[i + 3, j + 2] += t3 * c2
                D[i + 3, j + 3] += t3 * c3

# # ---------------------------------------------------------
# # Baseline 7: Blocked Parallel I using np.dot for Sub-blocks
# # ---------------------------------------------------------
# @njit(parallel=True, fastmath=True)
# def _2mm_blocked_np_dot_7(alpha, beta, A, B, C, D):
#     n = A.shape[0]
#     bs = BLOCK_SIZE
#     temp = np.zeros((n, n))

#     num_i_blocks = (n + bs - 1) // bs

#     for b in prange(num_i_blocks):
#         i_block = b * bs
#         i_end = min(i_block + bs, n)

#         for k_block in range(0, n, bs):
#             k_end = min(k_block + bs, n)
#             for j_block in range(0, n, bs):
#                 j_end = min(j_block + bs, n)
#                 temp[i_block:i_end, j_block:j_end] += alpha * np.dot(
#                     A[i_block:i_end, k_block:k_end],
#                     B[k_block:k_end, j_block:j_end],
#                 )

#         for j_block in range(0, n, bs):
#             j_end = min(j_block + bs, n)
#             D[i_block:i_end, j_block:j_end] *= beta

#         for k_block in range(0, n, bs):
#             k_end = min(k_block + bs, n)
#             for j_block in range(0, n, bs):
#                 j_end = min(j_block + bs, n)
#                 D[i_block:i_end, j_block:j_end] += np.dot(
#                     temp[i_block:i_end, k_block:k_end],
#                     C[k_block:k_end, j_block:j_end],
#                 )
                

# # ---------------------------------------------------------
# # Baseline 7: Blocked Temp Copy-In / Copy-Out
# # ---------------------------------------------------------
# @njit(parallel=True, fastmath=True)
# def _2mm_blocked_temp_copy_7(alpha, beta, A, B, C, D):
#     n = A.shape[0]
#     bs = BLOCK_SIZE
#     temp = np.zeros((n, n))

#     num_i_blocks = (n + bs - 1) // bs

#     for b in prange(num_i_blocks):
#         i_block = b * bs
#         i_end = min(i_block + bs, n)

#         for j_block in range(0, n, bs):
#             j_end = min(j_block + bs, n)

#             temp_copy = temp[i_block:i_end, j_block:j_end].copy()
            
#             for k_block in range(0, n, bs):
#                 k_end = min(k_block + bs, n)
#                 temp_copy += alpha * np.dot(
#                     A[i_block:i_end, k_block:k_end],
#                     B[k_block:k_end, j_block:j_end],
#                 )

#             temp[i_block:i_end, j_block:j_end] = temp_copy
            
#         for j_block in range(0, n, bs):
#             j_end = min(j_block + bs, n)
#             d_copy = (beta * D[i_block:i_end, j_block:j_end]).copy()
#             for k_block in range(0, n, bs):
#                 k_end = min(k_block + bs, n)
#                 d_copy += np.dot(
#                     temp[i_block:i_end, k_block:k_end],
#                     C[k_block:k_end, j_block:j_end],
#                 )
#             D[i_block:i_end, j_block:j_end] = d_copy


# # ---------------------------------------------------------
# # Baseline 8: Two-Level Blocked Parallel I with Temp Copy & np.dot
# # ---------------------------------------------------------
# @njit(parallel=True, fastmath=True)
# def _2mm_two_level_blocked_temp_np_dot_8(alpha, beta, A, B, C, D):
#     n = A.shape[0]
#     l2 = BLOCK_SIZE_L2
#     l1 = BLOCK_SIZE_L1
#     temp = np.zeros((n, n))

#     num_l2_i_blocks = (n + l2 - 1) // l2

#     for b2 in prange(num_l2_i_blocks):
#         i2_start = b2 * l2
#         i2_end = min(i2_start + l2, n)

#         for j2_start in range(0, n, l2):
#             j2_end = min(j2_start + l2, n)

#             temp_l2 = temp[i2_start:i2_end, j2_start:j2_end].copy()

#             for k2_start in range(0, n, l2):
#                 k2_end = min(k2_start + l2, n)

#                 for i1_start in range(i2_start, i2_end, l1):
#                     i1_end = min(i1_start + l1, i2_end)
#                     i1_rel_start = i1_start - i2_start
#                     i1_rel_end = i1_end - i2_start

#                     for j1_start in range(j2_start, j2_end, l1):
#                         j1_end = min(j1_start + l1, j2_end)
#                         j1_rel_start = j1_start - j2_start
#                         j1_rel_end = j1_end - j2_start

#                         temp_l1 = temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end].copy()

#                         for k1_start in range(k2_start, k2_end, l1):
#                             k1_end = min(k1_start + l1, k2_end)

#                             temp_l1 += alpha * np.dot(
#                                 A[i1_start:i1_end, k1_start:k1_end],
#                                 B[k1_start:k1_end, j1_start:j1_end],
#                             )

#                         temp_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end] = temp_l1

#             temp[i2_start:i2_end, j2_start:j2_end] = temp_l2

#         for j2_start in range(0, n, l2):
#             j2_end = min(j2_start + l2, n)

#             d_l2 = (beta * D[i2_start:i2_end, j2_start:j2_end]).copy()

#             for k2_start in range(0, n, l2):
#                 k2_end = min(k2_start + l2, n)

#                 for i1_start in range(i2_start, i2_end, l1):
#                     i1_end = min(i1_start + l1, i2_end)
#                     i1_rel_start = i1_start - i2_start
#                     i1_rel_end = i1_end - i2_start

#                     for j1_start in range(j2_start, j2_end, l1):
#                         j1_end = min(j1_start + l1, j2_end)
#                         j1_rel_start = j1_start - j2_start
#                         j1_rel_end = j1_end - j2_start

#                         d_l1 = d_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end].copy()

#                         for k1_start in range(k2_start, k2_end, l1):
#                             k1_end = min(k1_start + l1, k2_end)

#                             d_l1 += np.dot(
#                                 temp[i1_start:i1_end, k1_start:k1_end],
#                                 C[k1_start:k1_end, j1_start:j1_end],
#                             )

#                         d_l2[i1_rel_start:i1_rel_end, j1_rel_start:j1_rel_end] = d_l1

#             D[i2_start:i2_end, j2_start:j2_end] = d_l2

# # ---------------------------------------------------------
# # Baseline 9: Zero Allocation Blocked 2mm
# # ---------------------------------------------------------
# @njit(parallel=True, fastmath=True)
# def _2mm_blocked_zero_alloc_9(alpha, beta, A, B, C, D):
#     n = A.shape[0]
#     bs = BLOCK_SIZE
#     AB = np.zeros((n, n))

#     num_i_blocks = (n + bs - 1) // bs

#     for b in prange(num_i_blocks):
#         i_block = b * bs
#         i_end = min(i_block + bs, n)
#         h = i_end - i_block

#         temp_tile = np.empty((bs, bs), dtype=A.dtype)
#         for j_block in range(0, n, bs):
#             j_end = min(j_block + bs, n)
#             w = j_end - j_block

#             temp = temp_tile[:h, :w]
#             temp.fill(0.0)

#             for k_block in range(0, n, bs):
#                 k_end = min(k_block + bs, n)

#                 temp += alpha * np.dot(
#                     A[i_block:i_end, k_block:k_end],
#                     B[k_block:k_end, j_block:j_end]
#                 )

#             AB[i_block:i_end, j_block:j_end] = temp
        
#         for j_block in range(0, n, bs):
#             j_end = min(j_block + bs, n)
#             w = j_end - j_block
            
#             d_copy = (beta * D[i_block:i_end, j_block:j_end]).copy()
#             temp = d_copy[:h, :w]

#             for k_block in range(0, n, bs):
#                 k_end = min(k_block + bs, n)
#                 temp += np.dot(
#                     AB[i_block:i_end, k_block:k_end],
#                     C[k_block:k_end, j_block:j_end],
#                 )

#             D[i_block:i_end, j_block:j_end] = temp


# ---------------------------------------------------------
# Baseline 10: Reference NumPy dot and BLAS
# ---------------------------------------------------------
def _2mm_np_dot_10(alpha, beta, A, B, C, D):
    D[:] = alpha * np.dot(np.dot(A, B), C) + beta * D

def _2mm_AT_10(alpha, beta, A, B, C, D):
    D[:] = alpha * ((A @ B) @ C) + beta * D

def _2mm_np_matmul_10(alpha, beta, A, B, C, D):
    D[:] = alpha * np.matmul(np.matmul(A, B), C) + beta * D


# ---------------------------------------------------------
# Baseline 11: BLAS with block
# ---------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _2mm_np_dot_block_11(alpha, beta, A, B, C, D):
    mr = BLOCK_SIZE
    n = A.shape[0]
    for task in prange((n + mr - 1) // mr):
        i0 = task * mr
        i1 = min(i0 + mr, n)
        h = i1 - i0
        buf = np.empty((2, h, n), dtype=A.dtype)
        tmp = buf[0]
        out = buf[1]

        for i in prange(i0, i1):
            for j in prange(n):
                A[i, j] *= alpha

        np.dot(A[i0:i1], B, tmp)
        np.dot(tmp, C, out)

        for i in prange(i0, i1):
            for j in prange(n):
                D[i, j] = beta * D[i, j] + out[i - i0, j]


@njit(parallel=True, fastmath=True)
def _2mm_np_matmul_block_11(alpha, beta, A, B, C, D):
    mr = BLOCK_SIZE
    n = A.shape[0]
    for task in prange((n + mr - 1) // mr):
        i0 = task * mr
        i1 = min(i0 + mr, n)

        for i in prange(i0, i1):
            for j in prange(n):
                A[i, j] *= alpha

        temp = A[i0:i1] @ B
        out = temp @ C

        for i in prange(i0, i1):
            for j in prange(n):
                D[i, j] = beta * D[i, j] + out[i - i0, j]




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

    A_org = A.copy()
    D_org = D.copy()

    # Compute ground truth reference solution for correctness verification
    D_expected = alpha * np.dot(np.dot(A, B), C) + beta * D
    total_flops = 4.0 * (N ** 3) + 2.0 * (N ** 2) # 4N^3 + 2N^2

    def measure(fn, warmup=True, reps=9):
        if warmup:
            A[:] = A_org
            D[:] = D_org
            fn(alpha, beta, A, B, C, D)

        # Correctness check against reference matrix D_expected.
        # A is restored as well: some kernels scale it in place.
        A[:] = A_org
        D[:] = D_org
        fn(alpha, beta, A, B, C, D)
        is_correct = np.allclose(D, D_expected, rtol=1e-5, atol=1e-5)

        start = time.perf_counter()
        for _ in range(reps):
            A[:] = A_org
            D[:] = D_org
            fn(alpha, beta, A, B, C, D)
        elapsed = (time.perf_counter() - start) / reps
        gflops = (total_flops / elapsed) / 1e9

        return gflops, elapsed, is_correct

    results = []

    # 0. Pure Python, run at the full N so its GFLOP/s and speedups share total_flops.
    # Runtime grows as N^3 (roughly 2 minutes at N=512).
    A[:] = A_org
    D[:] = D_org

    t0 = time.perf_counter()
    _2mm_python_0(alpha, beta, A, B, C, D)
    py_elapsed = time.perf_counter() - t0
    py_gflops = (total_flops / py_elapsed) / 1e9
    py_correct = np.allclose(D, D_expected, rtol=1e-5, atol=1e-5)

    results.append(("0_python_naive", py_gflops, py_elapsed, py_correct))

    # Baseline functions list
    functions = [
        ("1_numba_naive", _2mm_numba_1),
        ("2_ijk_ijk", _2mm_ijk_ijk_2),
        ("2_ijk_ikj", _2mm_ijk_ikj_2),
        ("2_ijk_jik", _2mm_ijk_jik_2),
        # ("2_ijk_jki", _2mm_ijk_jki_2),
        # ("2_ijk_kij", _2mm_ijk_kij_2),
        # ("2_ijk_kji", _2mm_ijk_kji_2),
        # ("2_ikj_ijk", _2mm_ikj_ijk_2),
        ("2_ikj_ikj", _2mm_ikj_ikj_2),
        # ("2_ikj_jik", _2mm_ikj_jik_2),
        # ("2_ikj_jki", _2mm_ikj_jki_2),
        ("2_ikj_kij", _2mm_ikj_kij_2),
        # ("2_ikj_kji", _2mm_ikj_kji_2),
        # ("2_jik_ijk", _2mm_jik_ijk_2),
        # ("2_jik_ikj", _2mm_jik_ikj_2),
        # ("2_jik_jik", _2mm_jik_jik_2),
        # ("2_jik_jki", _2mm_jik_jki_2),
        # ("2_jik_kij", _2mm_jik_kij_2),
        # ("2_jik_kji", _2mm_jik_kji_2),
        # ("2_jki_ijk", _2mm_jki_ijk_2),
        # ("2_jki_ikj", _2mm_jki_ikj_2),
        # ("2_jki_jik", _2mm_jki_jik_2),
        # ("2_jki_jki", _2mm_jki_jki_2),
        # ("2_jki_kij", _2mm_jki_kij_2),
        # ("2_jki_kji", _2mm_jki_kji_2),
        # ("2_kij_ijk", _2mm_kij_ijk_2),
        # ("2_kij_ikj", _2mm_kij_ikj_2),
        # ("2_kij_jik", _2mm_kij_jik_2),
        # ("2_kij_jki", _2mm_kij_jki_2),
        # ("2_kij_kij", _2mm_kij_kij_2),
        # ("2_kij_kji", _2mm_kij_kji_2),
        # ("2_kji_ijk", _2mm_kji_ijk_2),
        # ("2_kji_ikj", _2mm_kji_ikj_2),
        # ("2_kji_jik", _2mm_kji_jik_2),
        # ("2_kji_jki", _2mm_kji_jki_2),
        # ("2_kji_kij", _2mm_kji_kij_2),
        # ("2_kji_kji", _2mm_kji_kji_2),
        ("3_fusion_ikj", _2mm_fusion_ikj_3),
        ("4_opt_flags_fusion_ikj", _2mm_opt_flags_fusion_ikj_4),
        ("5_parallel_i", _2mm_parallel_i_5),
        ("5_parallel_fusion_ikj", _2mm_parallel_fusion_ikj_5),
        ("6_blocked_parallel_i",_2mm_blocked_parallel_6),
        ("6_blocked_parallel_fusion_ikj", _2mm_parallel_fusion_blocked_6),
        ("8_loop_unrolling", _2mm_loop_unrolling_8),
        ("9_micro_kernel_2x2", _2mm_micro_kernel_2x2_9),
        ("9_micro_kernel_4x4", _2mm_micro_kernel_4x4_9),
        ("10_np_dot", _2mm_np_dot_10),
        ("10_AT", _2mm_AT_10),
        ("10_np_matmul", _2mm_np_matmul_10),
        ("11_np_dot_block", _2mm_np_dot_block_11),
        ("11_np_matmul_block", _2mm_np_matmul_block_11),
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