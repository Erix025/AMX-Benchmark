#include <chrono>
#include <iostream>
#include <string>

#include "amx.h"
#include "benchmark.h"
#include "utils.h"
namespace AMXBench {
void print_result(const std::string& name, const int iters,
                  const std::chrono::microseconds duration,
                  const long long ops_per_iter) {
    auto duration_per_iter = duration.count() / (double)iters;
    std::cout << name << ": " << duration_per_iter << " us" << std::endl;
    std::cout << "Throughput: " << ops_per_iter / duration_per_iter / 1e6
              << " TFLOPS" << std::endl;
}
namespace gemm {
void init_bench(int M, int K, int N, BF16* A, BF16* B, FP32* C) {
    random_buffer(A, M, K);
    random_buffer(B, K, N);
    init_buffer(C, (FP32)0, M, N);
}

void init_bench_parallel(int M, int K, int N, BF16* A, BF16* B, FP32* C,
                         int thread_row, int thread_col) {
    init_bench(M, K, N, A, B, C);
}
}  // namespace gemm
}  // namespace AMXBench