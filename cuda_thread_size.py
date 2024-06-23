"""
Time vs number of threads.
"""
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from time import perf_counter_ns
from matplotlib import pyplot as plt


@cuda.jit
def add_array_cuda(a: DeviceNDArray, b: DeviceNDArray, c: DeviceNDArray) -> None:
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if i < a.size:
        c[i] = a[i] + b[i]


def add_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + b


if __name__ == '__main__':
    N = 512*10000
    numberTimings = 101
    threadSizes = np.arange(8, 256)

    # Variables CPU
    a = np.arange(N, dtype=np.float32)
    b = np.arange(N, dtype=np.float32)
    c = np.empty(N, dtype=np.float32)

    # Variables GPU
    dev_a = cuda.to_device(a)
    dev_b = cuda.to_device(b)
    dev_c = cuda.device_array_like(a)

    # Compile and then clear GPU from tasks
    add_array_cuda[(N + 511) // 512, 512](dev_a, dev_b, dev_c)
    cuda.synchronize()

    # Timing GPU
    timingOne = np.empty(len(threadSizes))
    timingTwo = np.empty(numberTimings)
    for i in range(timingOne.size):
        threadsPerBlock = threadSizes[i]
        blocksPerGrid = (N + (threadsPerBlock - 1)) // threadsPerBlock
        for j in range(timingTwo.size):
            tic = perf_counter_ns()
            add_array_cuda[blocksPerGrid, threadsPerBlock](dev_a, dev_b, dev_c)
            cuda.synchronize()
            toc = perf_counter_ns()
            timingTwo[j] = toc - tic
        timingTwo *= 1e-3  # convert to μs
        timingOne[i] = np.mean(timingTwo)
    
    # Timing CPU
    timing = np.empty(numberTimings)
    for i in range(timing.size):
        tic = perf_counter_ns()
        c = add_array(a, b)
        toc = perf_counter_ns()
        timing[i] = toc - tic
    timing *= 1e-3  # convert to μs

    plt.plot(threadSizes, timingOne)
    plt.axhline(y=np.mean(timing), color='red')
    plt.legend(['CUDA', 'CPU'])
    plt.grid()
    plt.xlabel('Number Threads [-]')
    plt.ylabel('Mean Time [μs]')
    plt.savefig('plot_time_numberThreads')
    plt.show()