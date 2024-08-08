#include "utils.h"

#include <iostream>
#include <random>

#include "amx.h"

template <typename T>
void init_buffer(T *buf, T value, int32_t rows, int32_t cols) {
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      buf[i * cols + j] = value;
    }
}

void print_buffer(COMPLEX *buf, int32_t rows, int32_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << +buf[i * cols + j].real << "+" << +buf[i * cols + j].image
                << " ";
    }
    printf("\n");
  }
}

template <typename T>
void print_buffer(T *buf, int32_t rows, int32_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << +buf[i * cols + j] << " ";
    }
    printf("\n");
  }
  printf("\n");
}

template <typename T>
void random_buffer(T *buf, int32_t rows, int32_t cols) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // 定义浮点数的范围
  double min_range = 0.0;   // 范围的下限
  double max_range = 10.0;  // 范围的上限

  // 创建uniform_real_distribution对象
  std::uniform_real_distribution<> distrib(min_range, max_range);

  // 生成随机浮点数
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++) {
      buf[i * cols + j] = static_cast<T>(distrib(gen));
    }
}

template <typename T>
bool compare_buffer(T *buf1, T *buf2, int32_t rows, int32_t cols, double tol) {
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