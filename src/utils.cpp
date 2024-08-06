#include "utils.h"

#include <iostream>

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