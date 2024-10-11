#include <chrono>

#include "amx.h"
namespace AMXBench {
void tmul_benchmark_int8_uu();
void tmul_benchmark_int8_us();
void tmul_benchmark_int8_su();
void tmul_benchmark_int8_ss();
void tmul_benchmark_bf16(int M, int N, int K);
void tload_benchmark();
void tstore_benchmark();
void benchmark_all();
void benchmark_bf16_shapes();

// benchmark utils
void print_result(const std::string& name, const int iters,
                  const std::chrono::microseconds duration,
                  const long long ops_per_iter);
namespace gemm {
void init_bench(int M, int K, int N, BF16* A, BF16* B, FP32* C);
void init_bench_parallel(int M, int K, int N, BF16* A, BF16* B, FP32* C,
                         int thread_row, int thread_col);
}  // namespace gemm

// gemm benchmark
namespace gemm {
void bench_gemm_baseline(const int M, const int K, const int N, int max_iter);
void bench_gemm_reordered(const int M, const int K, const int N, int max_iter);
void bench_gemm_parallel(const int M, const int K, const int N, int max_iter,
                         int thread_row, int thread_col);
}  // namespace gemm
}  // namespace AMXBench