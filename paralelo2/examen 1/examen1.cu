#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>


__global__ void OrOperation(unsigned char *A, unsigned char *B, unsigned char *C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] | b[tid];
    }
}

int main()
{
    const int size_in_bytes = 1024;
    const int N_Size = size_in_bytes / sizeof(unsigned char);
    int threadsPerBlock = 1024; // Usual número de threads por bloque
    int blocksPerGrid = (N_Size + threadsPerBlock - 1) / threadsPerBlock;

    unsigned char *a_cpu, *b_cpu, *c_cpu;
    unsigned char *a_device, *b_device, *c_device;

    //memory allocation
    a_cpu = (unsigned char*)malloc(size_in_bytes);
    b_cpu = (unsigned char*)malloc(size_in_bytes);
    c_cpu = (unsigned char*)malloc(size_in_bytes);

    // device
    cudaMalloc((void**)&a_device, size_in_bytes);
    cudaMalloc((void**)&b_device, size_in_bytes);
    cudaMalloc((void**)&c_device, size_in_bytes);

    //transferir cpu host a gpu device
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);

    //lanzar kernel
    OrOperation <<<blocksPerGrid, threadsPerBlock>>>(a_device, b_device, c_device, size_in_bytes);

    for (int i = 0; i < 10; i++) {
        printf("A[%d] = %u, B[%d] = %u, C[%d] = A | B = %u\n", i, a_cpu[i], i, b_cpu[i], i, c_cpu[i]);
    }

    //transferir cpu device a gpu host
    cudaMemcpy(c_cpu, c_device, size_in_bytes, cudaMemcpyDeviceToHost);

    //liberar memoria
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    return 0;
}