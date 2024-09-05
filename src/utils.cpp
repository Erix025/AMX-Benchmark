#include "utils.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "amx.h"

void print_buffer(COMPLEX* buf, int32_t rows, int32_t cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << static_cast<double>(buf[i * cols + j].real) << "+"
                << static_cast<double>(buf[i * cols + j].image) << " ";
    }
    printf("\n");
  }
}

void bind_core(const int num_threads) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);  // clear cpuset
  // add threads to cpuset
  for (int i = 0; i < num_threads; i++) {
    CPU_SET(i, &cpuset);
  }

  // get pid
  pid_t pid = getpid();

  // bind threads to core
  if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) != 0) {
    perror("sched_setaffinity");
    return;
  }
  // std::cout << "Bind threads to cores successfully." << std::endl;
}

void bind_core(const std::vector<int>& thread_list) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);  // clear cpuset
  // add threads to cpuset
  for (auto i : thread_list) {
    CPU_SET(i, &cpuset);
  }

  // get pid
  pid_t pid = getpid();

  // bind threads to core
  if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) != 0) {
    perror("sched_setaffinity");
    return;
  }
  // std::cout << "Bind threads to cores successfully." << std::endl;
}

void reorder_matrix(BF16* A, const int row, const int col, const int tile_row,
                    const int tile_col) {
  BF16* A_copy = new BF16[row * col];
  memcpy(A_copy, A, sizeof(BF16) * row * col);
  const int tile_size = tile_row * tile_col;
  const int tile_row_num = row / tile_row;
  const int tile_col_num = col / tile_col;
  int target_offset = 0;
  for (int i = 0; i < tile_row_num; i += 2) {
    for (int j = 0; j < tile_col_num; j += 2) {
      for (int block_i = 0; block_i < 2; block_i++) {
        for (int block_j = 0; block_j < 2; block_j++) {
          int origin_offset =
              (i + block_i) * tile_row * col + (j + block_j) * tile_col;
          for (int ii = 0; ii < tile_row; ii++) {
            for (int jj = 0; jj < tile_col; jj++) {
              A[target_offset + ii * tile_col + jj] =
                  A_copy[origin_offset + ii * col + jj];
            }
          }
          target_offset += tile_size;
        }
      }
    }
  }
}