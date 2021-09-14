#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define RUNTIME_API_CALL(apiFuncCall)                                      \
{                                                                        \
  cudaError_t _status = apiFuncCall;                                     \
  if (_status != cudaSuccess) {                                          \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
      __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));    \
    exit(EXIT_FAILURE);                                                  \
  }                                                                      \
}

static inline double gettime_ms() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return (t.tv_sec+t.tv_nsec*1e-9)*1000;
}

__global__ void vectorAdd(const float *A, const float *B, float *C, uint64_t numElements) {
  uint64_t total_threads = blockDim.x * gridDim.x;
  uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  for( uint64_t tid = gid; tid < numElements; tid+=total_threads ) {
    if (tid < numElements) {
      C[tid] = A[tid] + B[tid];
    }
  }
}

int main(int argc, char *argv[]) {
  if(argc != 2) {
    fprintf(stderr, "usage: %s <numElements>\n", argv[0]);
    return (EXIT_FAILURE);
  }

  uint64_t numElements = atoll(argv[1]);
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %lu elements]\n", numElements);

  RUNTIME_API_CALL(cudaSetDevice(0));

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  float *d_A;
  float *d_B;
  float *d_C;

  RUNTIME_API_CALL(cudaMalloc(&d_A, size));
  RUNTIME_API_CALL(cudaMalloc(&d_B, size));
  RUNTIME_API_CALL(cudaMalloc(&d_C, size));

  //printf("&a 0x%" PRIx64 "\n", d_A);
  //printf("&b 0x%" PRIx64 "\n", d_B);
  //printf("&c 0x%" PRIx64 "\n", d_C);

  time_t t;
  srand((unsigned) time(&t));

  for (uint64_t i = 0; i < numElements; ++i) {
    h_A[i] = rand();
    h_B[i] = rand();
    h_C[i] = 0;
  }

  RUNTIME_API_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
  RUNTIME_API_CALL(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

  int threadsPerBlock = 64;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  double kernel_start = gettime_ms();
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  // vectorAdd<<< 48, 128 >>>(d_A, d_B, d_C, numElements);
  RUNTIME_API_CALL(cudaStreamSynchronize(0));
  double kernel_stop = gettime_ms();

  cudaError_t err = cudaSuccess;
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
    return EXIT_FAILURE;
  }

  RUNTIME_API_CALL(cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost));
  RUNTIME_API_CALL(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));
  RUNTIME_API_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

  // Verify that the result vector is correct
  for (uint64_t i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %ld!\n", i);
      return EXIT_FAILURE;
    }
  }

  fprintf(stderr, "vectorAdd duration: %lfms\n", (kernel_stop-kernel_start));
  printf("Test PASSED\n");

  free(h_A);
  free(h_B);
  free(h_C);

  RUNTIME_API_CALL(cudaFree(d_A));
  RUNTIME_API_CALL(cudaFree(d_B));
  RUNTIME_API_CALL(cudaFree(d_C));

  return EXIT_SUCCESS;
}

