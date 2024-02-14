#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<stdlib.h>


__global__ void print_all_idx()
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tidz = threadIdx.z;

    int bidx = blockIdx.x; 
    int bidy = blockIdx.y; 
    int bidz = blockIdx.z; 

    int gdimx = gridDim.x; 
    int gdimy = gridDim.y; 
    int gdimz = gridDim.z; 

    printf("[DEVICE] threadIdx.x :%d, blockIdx.x: %d, gridDim.x: %d \n", tidx, bidx, gdimx);
    printf("[DEVICE] threadIdx.y :%d, blockIdx.y: %d, gridDim.y: %d \n", tidy, bidy, gdimy);
    printf("[DEVICE] threadIdx.z :%d, blockIdx.z: %d, gridDim.z: %d \n", tidz, bidz, gdimz);
}

int main()
{
    //initialization
    dim3 blockSize(4, 4, 4);
    dim3 gridSize(2, 2, 2);

    int* c_cpu;
    int* a_cpu;
    int* b_cpu;

    int* c_device;
    int* a_device;
    int* b_device;
    const int data_count = 10000;
    const int data_size = data_count * sizeof(int);
    c_cpu = (int*)malloc(data_size);
    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);

    //memory allocation
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);

    //transfer cpu host to gpu device
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);

    //launch kernel
    print_all_idx << <gridSize, blockSize >> > ();

    cudaDeviceSynchronize();
    cudaDeviceReset();


    return 0;
}