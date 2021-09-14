#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

extern "C"
{
  #include <metamem_api.h>
  #include <metamem_pch.h>
}

#define RUNTIME_API_CALL(apiFuncCall)                                    \
{                                                                        \
  hipError_t _status = apiFuncCall;                                      \
  if (_status != hipSuccess) {                                           \
    fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
      __FILE__, __LINE__, #apiFuncCall, hipGetErrorString(_status));     \
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
  printf("[Vector addition of %lu elements]\n", numElements);

  RUNTIME_API_CALL(hipSetDevice(0));

  metamem* meta_h_A = metamem_init(METAMEM_HIP);
  metamem* meta_h_B = metamem_init(METAMEM_HIP);
  metamem* meta_h_C = metamem_init(METAMEM_HIP);

  meta_h_A->fn->alloc(meta_h_A, numElements, sizeof(float), MEM_CPU_PAGEABLE, MEM_GPU);
  meta_h_B->fn->alloc(meta_h_B, numElements, sizeof(float), MEM_CPU_PAGEABLE, MEM_GPU);
  meta_h_C->fn->alloc(meta_h_C, numElements, sizeof(float), MEM_CPU_PAGEABLE, MEM_GPU);

  float *h_A = (float *)meta_h_A->host_ptr->ptr;
  float *h_B = (float *)meta_h_B->host_ptr->ptr;
  float *h_C = (float *)meta_h_C->host_ptr->ptr;

  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  float *d_A = (float *)meta_h_A->device_ptr->ptr;
  float *d_B = (float *)meta_h_B->device_ptr->ptr;
  float *d_C = (float *)meta_h_C->device_ptr->ptr;

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

  meta_h_A->fn->copy(meta_h_A, H2D);
  meta_h_B->fn->copy(meta_h_B, H2D);
  meta_h_C->fn->copy(meta_h_C, H2D);

  int threadsPerBlock = 64;
  int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("HIP kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  double kernel_start = gettime_ms();
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  // vectorAdd<<< 48, 128 >>>(d_A, d_B, d_C, numElements);
  RUNTIME_API_CALL(hipStreamSynchronize(0));
  double kernel_stop = gettime_ms();

  hipError_t err = hipSuccess;
  err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", hipGetErrorString(err));
    return EXIT_FAILURE;
  }


  meta_h_A->fn->copy(meta_h_A, D2H);
  meta_h_B->fn->copy(meta_h_B, D2H);
  meta_h_C->fn->copy(meta_h_C, D2H);

  // Verify that the result vector is correct
  for (uint64_t i = 0; i < numElements; ++i) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %ld!\n", i);
      return EXIT_FAILURE;
    }
  }

  fprintf(stderr, "vectorAdd duration: %lfms\n", (kernel_stop-kernel_start));
  printf("Test PASSED\n");

  meta_h_A->fn->free(meta_h_A);
  meta_h_B->fn->free(meta_h_B);
  meta_h_C->fn->free(meta_h_C);

  metamem_shutdown(meta_h_A);
  metamem_shutdown(meta_h_B);
  metamem_shutdown(meta_h_C);

  return EXIT_SUCCESS;
}

