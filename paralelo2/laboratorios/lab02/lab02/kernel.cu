#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>

using namespace std;

#define GPUErrorAssertion(ans) {gpuAssert((ans), _FILE, __LINE_)}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: $s $s $d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel para sumar vectores
__global__ void sumVectors(int* a, int* b, int* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;
    int index = idx + idy * blockDim.x * gridDim.x + idz * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

    if (index < N) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int data_count = 10000;
    const int data_size = data_count * sizeof(int);

    // Punteros para el host
    int* c_cpu, * a_cpu, * b_cpu;
    // Punteros para el device
    int* c_device, * a_device, * b_device;

    // Asignación de memoria en el host
    c_cpu = (int*)malloc(data_size);
    a_cpu = (int*)malloc(data_size);
    b_cpu = (int*)malloc(data_size);

    // Inicialización de los vectores a y b
    for (int i = 0; i < data_count; i++) {
        a_cpu[i] = i;
        b_cpu[i] = i;
    }

    // Configuración del tamaño de bloque y de grid
    dim3 blockSize(4, 4, 4);
    dim3 gridSize((data_count + blockSize.x * blockSize.y * blockSize.z - 1) / (blockSize.x * blockSize.y * blockSize.z));

    //memory allocation
    cudaMalloc((void**)&c_device, data_size);
    cudaMalloc((void**)&a_device, data_size);
    cudaMalloc((void**)&b_device, data_size);

    // Transferencia de datos del device al host
    cudaMemcpy(c_device, c_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(a_device, a_cpu, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b_cpu, data_size, cudaMemcpyHostToDevice);

    // Lanzamiento del kernel
    sumVectors << <gridSize, blockSize >> > (a_device, b_device, c_device, data_count);

    // Mostrar el resultado de la suma
    for (int i = 0; i < data_count; i++) {
        printf("%d + %d = %d\n", a_cpu[i], b_cpu[i], c_cpu[i]);
    }

    // Liberación de memoria en el device
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    // Liberación de memoria en el host
    free(a_cpu);
    free(b_cpu);
    free(c_cpu);

    return 0;
}
