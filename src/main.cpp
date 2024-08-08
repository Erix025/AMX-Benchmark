//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
/* Initialize tile config */
#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemv.h"
#include "utils.cpp"
/* Initialize int8_t buffer */

int main() {
  int M = 32, N = 32;
  // FIXME: M = 101, N = 100 produce incorrect result.
  BF16* A = new BF16[M * N];
  BF16* x = new BF16[N];
  FP32* y1 = new FP32[M];
  FP32* y2 = new FP32[M];
  random_buffer(A, M, N);
  random_buffer(x, N, 1);
  init_buffer(y1, (FP32)0, M, 1);
  init_buffer(y2, (FP32)0, M, 1);
  gemv_ref(M, N, A, x, y2);
  gemv_naive(M, N, A, x, y1);
  if (compare_buffer(y1, y2, M, 1, 3)) {
    std::cout << "Identical result." << std::endl;
  } else {
    std::cout << "Wrong answer." << std::endl;
  }
  print_buffer(y1, 1, M);
  std::cout << "=================" << std::endl;
  print_buffer(y2, 1, M);
  return 0;
}
