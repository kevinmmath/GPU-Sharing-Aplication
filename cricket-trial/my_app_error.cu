#include <stdio.h>
#include <cuda_runtime.h>

/* ---------------------------------------------------------------
 * my_app_error.cu
 * Simulates erroneous host-to-client error returns via Cricket.
 *
 * Deliberately triggers three classes of CUDA errors:
 *   1. cudaMalloc with size=0            -> cudaErrorInvalidValue
 *   2. Kernel launch on NULL device ptr  -> cudaErrorIllegalAddress
 *   3. cudaMemcpy from NULL device ptr   -> cudaErrorInvalidValue
 *
 * Each error is caught and printed with cudaGetErrorString(), and
 * the error code is returned to main() -- simulating what Cricket
 * sends back from host to client over the RPC channel.
 * --------------------------------------------------------------- */

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t _e = (call);                                    \
        if (_e != cudaSuccess) {                                    \
            fprintf(stderr,                                         \
                "[HOST ERROR] %s:%d  '%s'  -> %s (code %d)\n",    \
                __FILE__, __LINE__, #call,                          \
                cudaGetErrorString(_e), (int)_e);                   \
            return (int)_e;                                         \
        }                                                           \
    } while (0)

__global__ void doubleArray(int *d_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        d_data[idx] = d_data[idx] * 2;
}

/* ---- Scenario 1: cudaMalloc with impossibly large size ---- */
int simulate_malloc_error(void) {
    printf("[HOST] Scenario 1: cudaMalloc(size=SIZE_MAX) -- bad allocation ...\n");
    int *d_bad = NULL;
    /* Requesting ~16 EB of device memory always fails */
    CUDA_CHECK(cudaMalloc((void **)&d_bad, (size_t)-1));
    cudaFree(d_bad);
    return 0;
}

/* ---- Scenario 2: kernel launched on NULL device pointer ---- */
int simulate_kernel_null_ptr(void) {
    printf("[HOST] Scenario 2: kernel on NULL device ptr ...\n");
    int *d_null = NULL;                     /* never allocated */
    doubleArray<<<1, 8>>>(d_null, 8);
    CUDA_CHECK(cudaDeviceSynchronize());    /* surfaces the async fault */
    return 0;
}

/* ---- Scenario 3: cudaMemcpy from NULL device pointer ---- */
int simulate_memcpy_null(void) {
    printf("[HOST] Scenario 3: cudaMemcpy from NULL device ptr ...\n");
    int h_data[8] = {0};
    CUDA_CHECK(cudaMemcpy(h_data, (void *)NULL,
                          8 * sizeof(int), cudaMemcpyDeviceToHost));
    return 0;
}

int main(void) {
    printf("[HOST] Starting CUDA error simulation.\n");
    printf("       Each cudaError_t is returned to the client via Cricket RPC.\n\n");

    int ret;

    ret = simulate_malloc_error();
    if (ret != 0)
        fprintf(stderr, "[CLIENT RECEIVES] Scenario 1 error code: %d\n\n", ret);
    cudaGetLastError();   /* reset sticky error state */

    ret = simulate_kernel_null_ptr();
    if (ret != 0)
        fprintf(stderr, "[CLIENT RECEIVES] Scenario 2 error code: %d\n\n", ret);
    cudaGetLastError();

    ret = simulate_memcpy_null();
    if (ret != 0)
        fprintf(stderr, "[CLIENT RECEIVES] Scenario 3 error code: %d\n\n", ret);
    cudaGetLastError();

    printf("[HOST] All error scenarios complete.\n");
    return 0;
}
