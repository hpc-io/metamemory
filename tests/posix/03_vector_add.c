#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <metamem_api.h>
#include <metamem_pch.h>

static inline double gettime_ms() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return (t.tv_sec+t.tv_nsec*1e-9)*1000;
}

void vectorAdd(const float *A, const float *B, float *C, uint64_t numElements) {
  for( uint64_t i = 0; i < numElements; i++ ) {
    C[i] = A[i] + B[i];
  }
}

int main(int argc, char *argv[]) {

  if(argc != 2) {
    fprintf(stderr, "usage: %s <numElements>\n", argv[0]);
    return (EXIT_FAILURE);
  }

  uint64_t numElements = atoll(argv[1]);
  printf("[Vector addition of %lu elements]\n", numElements);

  metamem* meta_h_A = metamem_init(METAMEM_POSIX);
  metamem* meta_h_B = metamem_init(METAMEM_POSIX);
  metamem* meta_h_C = metamem_init(METAMEM_POSIX);

  meta_h_A->fn->alloc(meta_h_A, numElements, sizeof(float), MEM_CPU_PAGEABLE, 0);
  meta_h_B->fn->alloc(meta_h_B, numElements, sizeof(float), MEM_CPU_PAGEABLE, 0);
  meta_h_C->fn->alloc(meta_h_C, numElements, sizeof(float), MEM_CPU_PAGEABLE, 0);

  float *A_ptr = meta_h_A->host_ptr->ptr;
  float *B_ptr = meta_h_B->host_ptr->ptr;
  float *C_ptr = meta_h_C->host_ptr->ptr;

  if (A_ptr == NULL || B_ptr == NULL ||C_ptr == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  time_t t;
  srand((unsigned) time(&t));

  for (uint64_t i = 0; i < numElements; ++i) {
    A_ptr[i] = rand();
    B_ptr[i] = rand();
    C_ptr[i] = 0;
  }

  double kernel_start = gettime_ms();
  vectorAdd(A_ptr, B_ptr, C_ptr, numElements);
  double kernel_stop = gettime_ms();

  // Verify that the result vector is correct
  for (uint64_t i = 0; i < numElements; ++i) {
    if (fabs(A_ptr[i] + B_ptr[i] - C_ptr[i]) > 1e-5) {
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

