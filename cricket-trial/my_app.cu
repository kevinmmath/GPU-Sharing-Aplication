#include <stdio.h>
#include <cuda_runtime.h>

__global__ void doubleArray(int *d_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_data[idx] = d_data[idx] * 2;
    }
}

int main() {
    const int N = 8;
    int h_data[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int *d_data = NULL;
    cudaMalloc((void **)&d_data, N * sizeof(int));
    cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice);
    doubleArray<<<1, 256>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Got back: ");
    for (int i = 0; i < N; i++) printf("%d ", h_data[i]);
    printf("\n");
    cudaFree(d_data);
    return 0;
}
