#include "amx.h"

// implementations of GEMV

void gemv_ref(const int M, const int N, const BF16* A, const BF16* x, FP32* y);

void gemv_naive(const int M, const int N, const BF16* A, const BF16* x,
                FP32* y);

void gemv_reordered(const int M, const int N, const BF16* A, const BF16* x,
                    FP32* y);

void multithread_gemv(const int num_threads, const int M, const int N,
                      const BF16* A, const BF16* x, FP32* y);

void benchmark_gemv(const int M, const int N, const int max_iter);

// utils for benchmark GEMV
void print_result(const std::string& name,
                  const std::chrono::microseconds duration, const int max_iter,
                  const int ops_per_iter);