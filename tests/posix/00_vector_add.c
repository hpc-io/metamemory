#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

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
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %lu elements]\n", numElements);

  float *h_A = (float *)malloc(size);
  float *h_B = (float *)malloc(size);
  float *h_C = (float *)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL)
  {
    fprintf(stderr, "Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  time_t t;
  srand((unsigned) time(&t));

  for (uint64_t i = 0; i < numElements; ++i) {
    h_A[i] = rand();
    h_B[i] = rand();
    h_C[i] = 0;
  }

  double kernel_start = gettime_ms();
  vectorAdd(h_A, h_B, h_C, numElements);
  double kernel_stop = gettime_ms();

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

  return EXIT_SUCCESS;
}

