//==============================================================
// Copyright © 2022 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
/* Initialize tile config */
#include "amx.h"
#include "examples.h"
#include "utils.cpp"
/* Initialize int8_t buffer */

int main() {
  tmul_int8_uu();
  tmul_int8_us();
  tmul_int8_su();
  tmul_int8_ss();
  return 0;
}
