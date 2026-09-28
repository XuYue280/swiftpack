import numpy as np
from swiftpack import swiftpack

from swiftpack.scheduler import AutoTileAndParallelizeScheduler, Scheduler



if __name__ == "__main__":
    N = 256
    alpha = np.float64(np.random.uniform(-1.0, 1.0))
    beta = np.float64(np.random.uniform(-1.0, 1.0))
    A = np.random.randn(N, N).astype(np.float64)
    B = np.random.randn(N, N).astype(np.float64)
    C = np.random.randn(N, N).astype(np.float64)
    D = np.random.randn(N, N).astype(np.float64)



    # Function decorated with custom scheduling strategy
    @swiftpack(scheduler=AutoTileAndParallelizeScheduler(), parallel=True, fastmath=True)
    def _2mm_swiftpack(alpha, beta, A, B, C, D):
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


    D_expected = alpha * np.dot(np.dot(A, B), C) + beta * D.copy()

    # Run transformed JIT function
    _2mm_swiftpack(alpha, beta, A, B, C, D)

    # Correctness check against standard matrix multiplication
    assert np.allclose(D, D_expected), "Verification Failed!"
    print("\nResult verified successfully against NumPy!")