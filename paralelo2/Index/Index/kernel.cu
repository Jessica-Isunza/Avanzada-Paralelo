
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void print_hello_cuda()
{
    int i = threadIdx.x;
    printf("[DEVICE] ThreadIdx: %d",i);
}

int main()
{
    print_hello_cuda << <1, 8 >> > ();
    return 0;
}