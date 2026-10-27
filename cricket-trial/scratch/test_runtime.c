#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount returned %d\n", err);
    if (err == cudaSuccess) {
        printf("Found %d devices\n", count);
    } else {
        printf("Error string: %s\n", cudaGetErrorString(err));
    }
    return 0;
}
