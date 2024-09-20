#include "gemm.h"

#include <chrono>
#include <iostream>
#include <string>

#include "amx.h"
#include "utils.h"

#define ENABLE_REF 1
#define ENABLE_NAIVE 1
#define ENABLE_REORDERED 0
#define ENABLE_PREFETCH 0
#define ENABLE_MULTITHREAD 0

namespace gemm {

void print_result(const std::string& name, const size_t iters,
                  const std::chrono::microseconds duration,
                  const size_t ops_per_iter) {
  auto duration_per_iter = duration.count() / (double)iters;
  std::cout << name << ": " << duration_per_iter << " us" << std::endl;
  std::cout << "Throughput: " << ops_per_iter / duration_per_iter / 1e6
            << " TFLOPS" << std::endl;
}

void _init_benchmark(size_t M, size_t K, size_t N, BF16* A, BF16* B, FP32* C) {
  random_buffer(A, M, K);
  random_buffer(B, K, N);
  init_buffer(C, (FP32)0, M, N);
}

void benchmark(const size_t M, const size_t K, const size_t N, int max_iter) {
  BF16* A = new BF16[M * K];
  BF16* B = new BF16[K * N];
  FP32* C0 = new FP32[M * N];
  FP32* C1 = new FP32[M * N];

  _init_benchmark(M, K, N, A, B, C0);
  init_buffer(C1, (FP32)0, M, N);

  size_t ops_per_iter = M * 2 * N * K;
  std::chrono::microseconds duration;

  bind_core(1);
#if ENABLE_REF
  // benchmark for reference version
  duration = measure_time<std::chrono::microseconds>(max_iter, gemm::reference,
                                                     M, K, N, A, B, C0);
  print_result("Reference version", max_iter, duration, ops_per_iter);
#endif
#if ENABLE_NAIVE
  // benchmark for naive version
  init_buffer(C1, (FP32)0, M, N);
  gemm::reorder_matrix_into_tile(B, K, N);
  duration = measure_time<std::chrono::microseconds>(max_iter, gemm::baseline,
                                                     M, K, N, A, B, C1);
  print_result("Naive version", max_iter, duration, ops_per_iter);
  compare_buffer_max(C0, C1, M, N, 10)
      ? std::cout << "Correctness: True" << std::endl
      : std::cout << "Correctness: False" << std::endl;
#endif
  // auto reorder_duration = measure_time<std::chrono::microseconds>(
  //     1, reorder_matrix, A, M, N, 16, 32);
  // std::cout << "Reorder matrix: " << reorder_duration.count() << " us"
  //           << std::endl;
#if ENABLE_REORDERED
#endif
#if ENABLE_PREFETCH
#endif
#if ENABLE_MULTITHREAD
#endif
  delete[] A;
  delete[] B;
  delete[] C0;
  delete[] C1;
}

void benchmark_gemv_with_preprocess(size_t M, size_t K, size_t N,
                                    int max_iter) {
  BF16* A = new BF16[M * K];
  BF16* B = new BF16[K * N];
  FP32* C0 = new FP32[M * N];
  FP32* C1 = new FP32[M * N];

  _init_benchmark(M, K, N, A, B, C0);
  init_buffer(C1, (FP32)0, M, N);

  size_t ops_per_iter = M * 2 * N * K;  // TODO: change to gemm ops
  std::chrono::microseconds duration;

  bind_core(1);
#if ENABLE_REF
  // benchmark for reference version
  duration = measure_time_with_preprocess<std::chrono::microseconds>(
      max_iter, gemm::reference, _init_benchmark, M, K, N, A, B, C0);
  print_result("Reference version", max_iter, duration, ops_per_iter);
#endif
#if ENABLE_NAIVE
  // benchmark for naive version
  init_buffer(C1, (FP32)0, M, N);
  gemm::reorder_matrix_into_tile(B, K, N);
  duration = measure_time_with_preprocess<std::chrono::microseconds>(
      max_iter, gemm::baseline, _init_benchmark, M, K, N, A, B, C1);
  print_result("Naive version", max_iter, duration, ops_per_iter);
  compare_buffer_max(C0, C1, M, N, 10)
      ? std::cout << "Correctness: True" << std::endl
      : std::cout << "Correctness: False" << std::endl;
#endif
#if ENABLE_REORDERED
#endif
#if ENABLE_PREFETCH
#endif
#if ENABLE_MULTITHREAD
#endif
  delete[] A;
  delete[] B;
  delete[] C0;
  delete[] C1;
}
}  // namespace gemm