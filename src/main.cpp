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
  // warm up
  benchmark_all();

  benchmark_all();
  return 0;
}
