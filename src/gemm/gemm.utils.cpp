#include "utils.h"

namespace gemm {
void reorder_matrix_into_tile(BF16 *A, const int row, const int col) {
  BF16 *A_copy = new BF16[row * col];
  std::memcpy(A_copy, A, row * col * sizeof(BF16));

  BF16 *row1 = A_copy;
  BF16 *row2 = A_copy + col;
  BF16 *A1 = A;
  for (int i = 0; i < row; i += 2) {
    for (int j = 0; j < col; j++) {
      A1[2 * j] = row1[j];
      A1[2 * j + 1] = row2[j];
    }
    A1 += 2 * col;
    row1 += 2 * col;
    row2 += 2 * col;
  }
  delete[] A_copy;
}
}  // namespace gemm