#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "amx.h"

template <typename T>
inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
void init_buffer(T *buf, T value, int32_t rows, int32_t cols) {
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      buf[i * cols + j] = value;
    }
}

template <typename T>
void print_buffer(T *buf, int32_t rows, int32_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << static_cast<double>(buf[i * cols + j]) << " ";
    }
    printf("\n");
  }
  printf("\n");
}

void print_buffer(COMPLEX *buf, int32_t rows, int32_t cols);

template <typename T>
void random_buffer(T *buf, int32_t rows, int32_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // 定义浮点数的范围
  double min_range = 0.0;   // 范围的下限
  double max_range = 10.0;  // 范围的上限

  // 创建uniform_real_distribution对象
  std::uniform_real_distribution<float> distrib(min_range, max_range);

  // 生成随机浮点数
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      buf[i * cols + j] = static_cast<T>(distrib(gen));
    }
}

template <typename T>
bool compare_buffer_l2norm(T *buf1, T *buf2, int32_t rows, int32_t cols,
                           double tol) {
  int i, j;
  double l2norm = 0;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      l2norm += (buf1[i * cols + j] - buf2[i * cols + j]) *
                (buf1[i * cols + j] - buf2[i * cols + j]);
    }
  l2norm = sqrt(l2norm);
  std::cout << "l2norm: " << l2norm << std::endl;
  return l2norm < tol;
}

template <typename T>
bool compare_buffer_max(T *buf1, T *buf2, int32_t rows, int32_t cols,
                        double tol) {
  int i, j;
  double max_norm = 0;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      max_norm = max(fabs(buf1[i * cols + j] - buf2[i * cols + j]), max_norm);
    }
  std::cout << "max norm: " << max_norm << std::endl;
  return max_norm < tol;
}

template <typename T>
void range_buffer(T *buf, int32_t rows, int32_t cols) {
  int i, j, value = 0;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      buf[i * cols + j] = value++;
    }
}

void bind_core(const int num_threads);
void bind_core(const std::vector<int> &thread_list);

void reorder_matrix(BF16 *A, const int row, const int col, const int tile_row,
                    const int tile_col);

template <typename Duration = std::chrono::microseconds, typename Func,
          typename... Args>
Duration measure_time(const uint iter, Func &&func, Args &&...args) {
  auto start = std::chrono::high_resolution_clock::now();
  for (uint i = 0; i < iter; i++) {
    // call the function with the arguments
    std::forward<Func>(func)(std::forward<Args>(args)...);
  }
  auto end = std::chrono::high_resolution_clock::now();
  // print time
  Duration duration = std::chrono::duration_cast<Duration>(end - start);
  return duration;
}

template <typename Duration = std::chrono::microseconds, typename Func,
          typename PreFunc, typename... Args>
Duration measure_time_with_preprocess(const uint iter, Func &&func,
                                      PreFunc &&pre_func, Args &&...args) {
  Duration duration(0);
  for (uint i = 0; i < iter; i++) {
    std::forward<PreFunc>(pre_func)(std::forward<Args>(args)...);
    // call the function with the arguments
    auto start = std::chrono::high_resolution_clock::now();
    std::forward<Func>(func)(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    auto _duration = std::chrono::duration_cast<Duration>(end - start);
    // std::cout << "iter " << i << ": " << _duration.count() << " us"
    //           << std::endl;
    duration += _duration;
  }
  // print time
  return duration;
}

namespace gemm {
void reorder_matrix_into_tile(BF16 *A, const int row, const int col);
}  // namespace gemm