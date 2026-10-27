/*
 * linear_regression_dataset.cu
 * ==============================
 * Reads a real dataset (dataset.csv) and trains a linear regression model
 * y = w*x + b via gradient descent, with ALL gradient computation happening
 * on the (remote) GPU through Cricket.
 *
 * IMPORTANT — how file I/O works with Cricket:
 *   Cricket's LD_PRELOAD only intercepts CUDA runtime calls (cudaMalloc,
 *   cudaMemcpy, kernel launches, etc). Regular file I/O (fopen/fread) is
 *   NOT intercepted — it always happens on whichever machine actually
 *   executes this binary. So dataset.csv must sit next to this binary
 *   wherever you run it (client or host).
 *
 * DATA FLOW:
 *   1. This program (running on your machine) reads dataset.csv from disk.
 *   2. x[] and y[] arrays are uploaded to GPU memory via cudaMemcpy —
 *      THIS call is intercepted by Cricket and sent to the host's GPU.
 *   3. Every gradient_reduce kernel launch executes on the host GPU.
 *   4. After training, the learned w, b, and predictions are copied back
 *      to this machine via cudaMemcpy (DeviceToHost) — again via Cricket.
 *
 * HOW TO BUILD
 *   nvcc -O2 -o linear_regression_dataset linear_regression_dataset.cu -lcudart
 *
 * HOW TO RUN via Cricket (dataset.csv must be in the same folder)
 *   REMOTE_GPU_ADDRESS=<host_ip> LD_PRELOAD=./cricket-client.so ./linear_regression_dataset
 *
 * HOW TO RUN LOCALLY (ground truth, no Cricket, run directly on host)
 *   ./linear_regression_dataset
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_SAMPLES 10000
#define EPOCHS      1000
#define LR          0.001f
#define BLOCK_SIZE  256

#define CUDA_CHECK(x) do { \
    cudaError_t e=(x); \
    if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} \
} while(0)

/* ── Load CSV: expects header "x,y" then one "x,y" pair per line ───────────
 * Runs as plain C file I/O — NOT intercepted by Cricket, executes locally
 * on whichever machine runs this binary.
 */
static int load_csv(const char *path, float *x, float *y, int max_n)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "ERROR: could not open '%s' — make sure dataset.csv\n", path);
        fprintf(stderr, "       is in the same directory as this executable.\n");
        exit(1);
    }

    char line[256];
    int n = 0;

    /* skip header */
    if (!fgets(line, sizeof(line), f)) { fclose(f); return 0; }

    while (n < max_n && fgets(line, sizeof(line), f)) {
        float xi, yi;
        if (sscanf(line, "%f,%f", &xi, &yi) == 2) {
            x[n] = xi;
            y[n] = yi;
            n++;
        }
    }
    fclose(f);
    return n;
}

/*
 * Kernel: compute per-sample gradient contributions and reduce them to
 * two scalars (sum of err*x, sum of err) using shared memory + atomicAdd
 * across blocks. Handles any N via grid-stride.
 */
__global__ void gradient_reduce(const float *x, const float *y,
                                 float w, float b,
                                 float *out, int n)
{
    __shared__ float s_dw[BLOCK_SIZE];
    __shared__ float s_db[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;

    float dw = 0.f, db = 0.f;
    if (i < n) {
        float err = (w * x[i] + b) - y[i];
        dw = err * x[i];
        db = err;
    }
    s_dw[t] = dw;
    s_db[t] = db;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            s_dw[t] += s_dw[t + stride];
            s_db[t] += s_db[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        atomicAdd(&out[0], s_dw[0]);
        atomicAdd(&out[1], s_db[0]);
    }
}

/* Kernel: final predictions with learned w, b */
__global__ void predict(const float *x, float w, float b, float *preds, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) preds[i] = w * x[i] + b;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void)
{
    /* ── 1. Load dataset (local file I/O — runs on whichever machine
     *      executes this binary, NOT routed through Cricket) ──────────── */
    float *h_x = (float*)malloc(MAX_SAMPLES * sizeof(float));
    float *h_y = (float*)malloc(MAX_SAMPLES * sizeof(float));
    int N = load_csv("dataset.csv", h_x, h_y, MAX_SAMPLES);

    if (N == 0) {
        fprintf(stderr, "ERROR: dataset.csv contained no valid rows\n");
        return 1;
    }

    printf("============================================================\n");
    printf("  Dataset loaded: %d samples from dataset.csv\n", N);
    printf("============================================================\n");
    printf("  First 5 rows:\n");
    for (int i = 0; i < 5 && i < N; i++)
        printf("    x=%.4f  y=%.4f\n", h_x[i], h_y[i]);
    printf("\n");

    /* ── 2. Upload to GPU — THIS is where Cricket sends data to the
     *      remote host's GPU memory ─────────────────────────────────── */
    float *d_x, *d_y, *d_out, *d_preds;
    CUDA_CHECK(cudaMalloc(&d_x,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,     N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,   2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_preds, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    /* ── 3. Train: gradient descent, all math on GPU ─────────────────── */
    printf("============================================================\n");
    printf("  Training on (remote) GPU — %d epochs, lr=%.4f\n", EPOCHS, LR);
    printf("============================================================\n");

    float w = 0.f, b = 0.f;
    float h_out[2];

    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);

    double t0 = now_ms();

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        CUDA_CHECK(cudaMemset(d_out, 0, 2 * sizeof(float)));

        gradient_reduce<<<grid, block>>>(d_x, d_y, w, b, d_out, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_out, d_out, 2 * sizeof(float), cudaMemcpyDeviceToHost));

        w -= LR * (h_out[0] / N);
        b -= LR * (h_out[1] / N);

        if ((epoch + 1) % 100 == 0)
            printf("  epoch %4d  w=%.4f  b=%.4f\n", epoch + 1, w, b);
    }

    double train_ms = now_ms() - t0;

    /* ── 4. Final predictions + metrics, computed on GPU, sent back ──── */
    predict<<<grid, block>>>(d_x, w, b, d_preds, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_preds = (float*)malloc(N * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_preds, d_preds, N * sizeof(float), cudaMemcpyDeviceToHost));

    float mse = 0.f;
    for (int i = 0; i < N; i++) {
        float err = h_y[i] - h_preds[i];
        mse += err * err;
    }
    mse /= N;

    /* ── 5. Results — this is what gets "sent back" to the client ────── */
    printf("\n============================================================\n");
    printf("  RESULTS (received from remote GPU via Cricket)\n");
    printf("============================================================\n");
    printf("  Learned weight (w) : %.6f\n", w);
    printf("  Learned bias   (b) : %.6f\n", b);
    printf("  Fitted line        : y = %.6f * x + %.6f\n", w, b);
    printf("  MSE                : %.6f\n", mse);
    printf("  RMSE               : %.6f\n", sqrtf(mse));
    printf("  Training duration  : %.2f ms (%d epochs, includes network latency)\n",
           train_ms, EPOCHS);
    printf("  Sample predictions :\n");
    printf("    %-10s %-10s %-10s\n", "x", "y_true", "y_pred");
    for (int i = 0; i < 5 && i < N; i++)
        printf("    %-10.4f %-10.4f %-10.4f\n", h_x[i], h_y[i], h_preds[i]);
    printf("============================================================\n\n");

    free(h_x); free(h_y); free(h_preds);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_out); cudaFree(d_preds);
    return 0;
}
