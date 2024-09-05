#include <chrono>
#include <iostream>
#include <string>

#include "amx.h"
#include "gemv.h"
#include "utils.h"

void print_result(const std::string& name,
                  const std::chrono::microseconds duration,
                  const int ops_per_iter) {
  std::cout << name << ": " << duration.count() << " us" << std::endl;
  std::cout << "Throughput: " << ops_per_iter / duration.count() / 1e6
            << " TFLOPS" << std::endl;
}

void benchmark_gemv(int M, int N, int max_iter) {
  BF16* A = new BF16[M * N];
  BF16* x = new BF16[N];
  FP32* y0 = new FP32[M];
  FP32* y1 = new FP32[M];

  random_buffer(A, M, N);
  random_buffer(x, N, 1);
  init_buffer(y0, (FP32)0, M, 1);  // y0 is the reference
  init_buffer(y1, (FP32)0, M, 1);  // y1 is the result for comparison

  int ops_per_iter = M * 2 * N;
  std::chrono::microseconds duration;

  // benchmark for reference version
  bind_core(1);
  duration = measure_time<std::chrono::microseconds>(max_iter, gemv_ref, M, N,
                                                     A, x, y0);
  print_result("Reference version", duration, ops_per_iter);

  // benchmark for naive version
  duration = measure_time<std::chrono::microseconds>(max_iter, gemv_naive, M, N,
                                                     A, x, y1);
  print_result("Naive version", duration, ops_per_iter);
  //   compare_buffer_max(y0, y1, M, 1, 10)
  //       ? std::cout << "Correctness: True" << std::endl
  //       : std::cout << "Correctness: False" << std::endl;

  // benchmark for reordered version
  init_buffer(y1, (FP32)0, M, 1);
  reorder_matrix(A, M, N, 16, 32);
  duration = measure_time<std::chrono::microseconds>(max_iter, gemv_reordered,
                                                     M, N, A, x, y1);
  print_result("Reordered version", duration, ops_per_iter);
  //   compare_buffer_max(y0, y1, M, 1, 10)
  //       ? std::cout << "Correctness: True" << std::endl
  //       : std::cout << "Correctness: False" << std::endl;

  // benchmark for multithread version
  int num_threads = 4;
  init_buffer(y1, (FP32)0, M, 1);
  bind_core(num_threads);
  duration = measure_time<std::chrono::microseconds>(
      max_iter, multithread_gemv, num_threads, M, N, A, x, y1);
  print_result("Multithread version", duration, ops_per_iter);
  //   compare_buffer_max(y0, y1, M, 1, 10)
  //       ? std::cout << "Correctness: True" << std::endl
  //       : std::cout << "Correctness: False" << std::endl;

  delete[] A;
  delete[] x;
  delete[] y0;
  delete[] y1;
}