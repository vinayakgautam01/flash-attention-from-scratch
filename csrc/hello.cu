// csrc/hello.cu
// M0 build sanity: proves nvcc works, links, and can launch a kernel.
// Also prints the GPU numbers we need to paste into docs/AGENTS.md
// (SM version, shared-mem/CTA, HBM bandwidth) before starting M9's roofline.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void greet() {
    printf("hello from GPU (block %d, thread %d)\n", blockIdx.x, threadIdx.x);
}

static void check(cudaError_t err, const char *where) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");

    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, device), "cudaGetDeviceProperties");

    // Peak HBM bandwidth (DDR): 2 * clock(Hz) * bus_width(bytes) / 1e9
    double bw_gb_s =
        2.0 * (static_cast<double>(p.memoryClockRate) * 1000.0) * (p.memoryBusWidth / 8.0) / 1.0e9;

    printf("device            : %s\n", p.name);
    printf("compute capability: %d.%d (arch sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
    printf("SMs               : %d\n", p.multiProcessorCount);
    printf("global memory     : %.2f GB\n",
           static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    printf("shared mem / SM   : %d KB\n", static_cast<int>(p.sharedMemPerMultiprocessor / 1024));
    printf("shared mem / CTA  : %d KB\n", static_cast<int>(p.sharedMemPerBlock / 1024));
    printf("registers / SM    : %d\n", p.regsPerMultiprocessor);
    printf("HBM clock         : %d MHz\n", p.memoryClockRate / 1000);
    printf("HBM bus width     : %d bits\n", p.memoryBusWidth);
    printf("peak HBM BW       : %.1f GB/s\n", bw_gb_s);

    greet<<<1, 4>>>();
    check(cudaDeviceSynchronize(), "greet sync");

    return 0;
}
