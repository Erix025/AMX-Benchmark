#include "amx.h"
#include "utils.h"

void gemv_ref(const int M, const int N, const BF16* A, const BF16* x, FP32* y) {
  for (int i = 0; i < M; i++) {
    FP32 sum = 0.0f;
    for (int j = 0; j < N; j++) {
      sum += A[i * N + j] * x[j];
    }
    y[i] = sum;
  }
}