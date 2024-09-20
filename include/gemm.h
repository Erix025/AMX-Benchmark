#pragma once
#include "amx.h"

namespace gemm {
void baseline(const int M, const int K, const int N, const BF16* A,
              const BF16* B, FP32* C);
void reference(const int M, const int K, const int N, const BF16* A,
               const BF16* B, FP32* C);
void benchmark(const size_t M, const size_t K, const size_t N, int max_iter);
}  // namespace gemm