#include <cuda.h>
#include <stdio.h>

int main() {
    CUresult res = cuInit(0);
    printf("cuInit(0) returned %d\n", res);
    if (res == CUDA_SUCCESS) {
        int count = 0;
        cuDeviceGetCount(&count);
        printf("Found %d devices\n", count);
    }
    return 0;
}
