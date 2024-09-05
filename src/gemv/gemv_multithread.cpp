#include <thread>
#include <vector>

#include "amx.h"
#include "gemv.h"
#include "utils.h"
void _thread_gemv(const int local_M, const int N, const BF16* local_A,
                  const BF16* x, FP32* local_y) {
  gemv_reordered(local_M, N, local_A, x, local_y);
}

void multithread_gemv(const int num_threads, const int M, const int N,
                      const BF16* A, const BF16* x, FP32* y) {
  std::vector<std::thread> threads;
  // launch threads
  for (int i = 0; i < num_threads; i++) {
    int local_M = M / num_threads;
    const BF16* local_A = A + i * local_M * N;
    FP32* local_y = y + i * local_M;
    // main thread
    if (i == num_threads - 1) {
      local_M = M - i * local_M;
      _thread_gemv(local_M, N, local_A, x, local_y);
      continue;
    }
    auto thread = std::thread(_thread_gemv, local_M, N, local_A, x, local_y);
    threads.push_back(std::move(thread));
  }
  // wait for all threads to finish
  for (auto& thread : threads) {
    thread.join();
  }
}