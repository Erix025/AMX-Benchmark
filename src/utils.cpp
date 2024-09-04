#include "utils.h"

#include <iostream>
#include <random>

#include "amx.h"

void print_buffer(COMPLEX *buf, int32_t rows, int32_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << static_cast<double>(buf[i * cols + j].real) << "+"
                << static_cast<double>(buf[i * cols + j].image) << " ";
    }
    printf("\n");
  }
}