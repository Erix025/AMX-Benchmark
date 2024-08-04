//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
/* Initialize tile config */
#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "utils.cpp"
/* Initialize int8_t buffer */

int main() {
  tload_benchmark_int8();
  tmul_benchmark_int8_uu();
  tstore_benchmark_int8();
  return 0;
}
