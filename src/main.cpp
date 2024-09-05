#include <chrono>

#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemv.h"
#include "sched.h"
#include "utils.h"
int main() {
  int M = 10240 * 4, N = 10240;
  benchmark_gemv(M, N, 100);
  return 0;
}
