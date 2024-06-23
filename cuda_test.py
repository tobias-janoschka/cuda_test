"""
23.06.2024
Adapted code from:
https://towardsdatascience.com/cuda-by-numba-examples-1-4-e0d06651612f
"""

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from time import perf_counter_ns


@cuda.jit
def add_array_cuda(a: DeviceNDArray, b: DeviceNDArray, c: DeviceNDArray) -> None:
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < a.size:
        c[i] = a[i] + b[i]


def add_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


if __name__ == '__main__':
    N = 512*10000
    numberTimings = 21

    # Variables CPU
    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty(N, dtype=np.float32)

    # Variables GPU
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)
    dev_c = cuda.device_array_like(a)
    threadsPerBlock = 64
    blocksPerGrid = (N + (threadsPerBlock - 1)) // threadsPerBlock
    print('blocksPerGrid: ' + str(blocksPerGrid) + '\tthreadsPerBlock:' + str(threadsPerBlock))

    # Compile and then clear GPU from tasks
    add_array_cuda[blocksPerGrid, threadsPerBlock](dev_a, dev_b, dev_c)
    cuda.synchronize()

    # Timing GPU
    timing = np.empty(numberTimings)
    for i in range(timing.size):
        tic = perf_counter_ns()
        add_array_cuda[blocksPerGrid, threadsPerBlock](dev_a, dev_b, dev_c)
        cuda.synchronize()
        toc = perf_counter_ns()
        timing[i] = toc - tic
    timing *= 1e-3  # convert to μs
    print(f"Elapsed time CUDA: {timing.mean():.0f} ± {timing.std():.0f} μs")

    # Timing CPU
    timing = np.empty(numberTimings)
    for i in range(timing.size):
        tic = perf_counter_ns()
        c = add_array(a, b)
        toc = perf_counter_ns()
        timing[i] = toc - tic
    timing *= 1e-3  # convert to μs
    print(f"Elapsed time CPU: {timing.mean():.0f} ± {timing.std():.0f} μs")

    """
    output:

    blocksPerGrid: 80000    threadsPerBlock:64
    Elapsed time CUDA: 1533 ± 42 μs
    Elapsed time CPU: 6978 ± 1100 μs
    """
