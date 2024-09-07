#include <chrono>

#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemv.h"
#include "sched.h"
#include "utils.h"
int main() {
  size_t M = 1024 * 96, N = 10240;
  benchmark_gemv(M, N, 1000);
  return 0;
}
