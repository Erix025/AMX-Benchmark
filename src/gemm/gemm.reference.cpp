#include "amx.h"
#include "utils.h"

namespace gemm {

/// @brief Reference implementation of GEMM.
/// @param M Number of rows of matrix A.
/// @param K Number of columns of matrix A and rows of matrix B.
/// @param N Number of columns of matrix B.
/// @param A Pointer to matrix A.
/// @param B Pointer to matrix B.
/// @param C Pointer to matrix C.

void reference(const int M, const int K, const int N, const BF16* A,
               const BF16* B, FP32* C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        C[i * N + j] += (FP32)A[i * K + k] * (FP32)B[k * N + j];
      }
    }
  }
}
}  // namespace gemm