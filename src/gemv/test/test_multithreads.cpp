#include "amx.h"
#include "gemv.h"
#include "utils.h"

void test_multithreads(const int M, const int N) {
  BF16* A = new BF16[M * N];
  BF16* x = new BF16[N];
  FP32* y = new FP32[M];

  for (int i = 1; i <= 16; i++) {
    bind_core(i);
    random_buffer(A, M, N);
    random_buffer(x, N, 1);
    init_buffer(y, (FP32)0, M, 1);
    auto duration = measure_time<std::chrono::microseconds>(
        1024, multithread_gemv, i, M, N, A, x, y);
    std::cout << "Number of threads: " << i << std::endl;
    print_result("Prefetch version", 1024, duration, M * 2 * N);
  }
}

