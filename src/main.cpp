#include <chrono>

#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemv.h"
#include "sched.h"
#include "utils.h"
int main() {
  size_t M = 4096, N = 4096;
  benchmark_gemv(M, N, 10000);
  // tmul_benchmark_bf16(16, 1, 16);
  // test_gemv_prefetch_offset(M, N);
  return 0;
}
