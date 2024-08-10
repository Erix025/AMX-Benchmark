#include <chrono>

#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemv.h"
#include "utils.cpp"
int main() {
  benchmark_gemv();
  return 0;
}
