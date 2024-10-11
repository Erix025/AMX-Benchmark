#pragma once
#include "amx.h"
namespace AMXBench {
namespace gemm {
void baseline(const int M, const int K, const int N, const BF16* A,
              const BF16* B, FP32* C);
void reference(const int M, const int K, const int N, const BF16* A,
               const BF16* B, FP32* C);
void reordered(const int M, const int K, const int N, const BF16* A,
               const BF16* B, FP32* C);
void parallel(const int M, const int K, const int N, const BF16* A,
              const BF16* B, FP32* C, const int thread_row,
              const int thread_col);

// utils
void reorder_matrix_into_tile(BF16* A, const int row, const int col);
void pack_matrix_A(const int M, const int K, BF16* A);
void pack_matrix_B(const int K, const int N, BF16* B);
}  // namespace gemm
}  // namespace AMXBench