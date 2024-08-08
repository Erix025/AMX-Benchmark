#include "gemv.h"

#include "amx.h"
#include "utils.cpp"
void gemv() {
  BF16 A[16 * 32];
  BF16 x[32];
  FP32 y[16];
  // init data
  init_buffer(A, (BF16)0.1, 16, 32);
  init_buffer(x, (BF16)1, 16, 2);
  init_buffer(y, (FP32)0, 16, 1);

  gemv_tile(A, x, y);

  print_buffer(y, 16, 1);
}
/**
 * Naive version of GEMV with AMX.
 * This version will use at most 3 tile registers to perform GEMV.
 * Please make sure that N is even.
 */
void gemv_naive(const int M, const int N, const BF16* A, const BF16* x,
                FP32* y) {
  // check if N is even
  if (N % 2) {
    std::cerr << "Error: N must be even." << std::endl;

    return;
  }
  // enable amx
  if (!enable_amx()) exit(-1);
  // init default config
  __tilecfg default_config = {0};
  {
    default_config.palette_id = 1;
    default_config.start_row = 0;
    // y
    default_config.rows[0] = 16;
    default_config.colsb[0] = 4;
    // A
    default_config.rows[1] = 16;
    default_config.colsb[1] = 64;
    // x
    default_config.rows[2] = 16;
    default_config.colsb[2] = 4;
  }
  // init row_tail config
  int row_tail = N % 32;
  // BF16* x_tail = nullptr;
  // // if row_tail is odd, do 0 padding
  // if (row_tail % 2 != 0) {
  //   x_tail = new BF16[row_tail + 1];
  //   const BF16* x_pointer = x + (N - row_tail);
  //   for (int i = 0; i < row_tail; i++) {
  //     x_tail[i] = x_pointer[i];
  //   }
  //   x_tail[row_tail] = 0;  // 0 padding
  // }
  __tilecfg row_tail_config = {0};
  {
    row_tail_config.palette_id = 1;
    row_tail_config.start_row = 0;
    // y
    row_tail_config.rows[0] = 16;
    row_tail_config.colsb[0] = 4;
    // A
    row_tail_config.rows[1] = 16;
    row_tail_config.colsb[1] = row_tail * sizeof(BF16);
    // x
    row_tail_config.rows[2] = row_tail / 2;
    row_tail_config.colsb[2] = 4;
  }
  // init col_tail config
  int col_tail = M % 16;
  __tilecfg col_tail_config = {0};
  {
    col_tail_config.palette_id = 1;
    col_tail_config.start_row = 0;
    // y
    col_tail_config.rows[0] = col_tail;
    col_tail_config.colsb[0] = 4;
    // A
    col_tail_config.rows[1] = col_tail;
    col_tail_config.colsb[1] = 64;
    // x
    col_tail_config.rows[2] = 16;
    col_tail_config.colsb[2] = 4;
  }
  // init corner config
  __tilecfg corner_config = {0};
  {
    corner_config.palette_id = 1;
    corner_config.start_row = 0;
    // y
    corner_config.rows[0] = col_tail;
    corner_config.colsb[0] = 4;
    // A
    corner_config.rows[1] = col_tail;
    corner_config.colsb[1] = row_tail * sizeof(BF16);
    // x
    corner_config.rows[2] = row_tail / 2;
    corner_config.colsb[2] = 4;
  }

  _tile_loadconfig(&default_config);
  int i, j;
  for (i = 0; i <= M - 16; i += 16) {
    _tile_loadd(0, y + i, sizeof(FP32));
    for (j = 0; j <= N - 32; j += 32) {
      _tile_loadd(1, A + i * N + j, sizeof(BF16) * N);
      _tile_loadd(2, x + j, 2 * sizeof(BF16));
      _tile_dpbf16ps(0, 1, 2);
    }
    if (j < N) {
      // for row tail
      _tile_stored(0, y + i, sizeof(FP32));
      _tile_loadconfig(&row_tail_config);
      _tile_loadd(0, y + i, sizeof(FP32));
      _tile_loadd(1, A + i * N + j, sizeof(BF16) * N);
      // if (x_tail == nullptr)
      _tile_loadd(2, x + j, 2 * sizeof(BF16));
      // else
      //   _tile_loadd(2, x_tail, 2 * sizeof(BF16));
      _tile_dpbf16ps(0, 1, 2);
    }
    _tile_stored(0, y + i, sizeof(FP32));
  }
  if (i < M) {
    // for col tail
    _tile_loadconfig(&col_tail_config);
    _tile_loadd(0, y + i, sizeof(FP32));
    for (j = 0; j <= N - 32; j += 32) {
      _tile_loadd(1, A + i * N + j, sizeof(BF16) * N);
      _tile_loadd(2, x + j, 2 * sizeof(BF16));
      _tile_dpbf16ps(0, 1, 2);
    }
    if (j < N) {
      // for corner
      _tile_stored(0, y + i, sizeof(FP32));
      _tile_loadconfig(&corner_config);
      _tile_loadd(0, y + i, sizeof(FP32));
      _tile_loadd(1, A + i * N + j, sizeof(BF16) * N);
      // if (x_tail == nullptr) {
      _tile_loadd(2, x + j, 2 * sizeof(BF16));
      // } else {
      //   _tile_loadd(2, x_tail, 2 * sizeof(BF16));
      // }
      _tile_dpbf16ps(0, 1, 2);
      _tile_stored(0, y + i, sizeof(FP32));
    }
  }
}

void gemv_tile(BF16* A, BF16* x, FP32* y) {
  // enable amx
  if (!enable_amx()) exit(-1);
  // init config
  __tilecfg config = {0};
  {
    config.palette_id = 1;
    config.start_row = 0;
    // y
    config.rows[0] = 16;
    config.colsb[0] = 4;
    // A
    config.rows[1] = 16;
    config.colsb[1] = 64;
    // x
    config.rows[2] = 16;
    config.colsb[2] = 4;
  }
  _tile_loadconfig(&config);
  // load
  _tile_loadd(0, y, 4);
  _tile_loadd(1, A, 64);
  _tile_loadd(2, x, 4);
  // compute
  _tile_dpbf16ps(0, 1, 2);
  // store
  _tile_stored(0, y, 4);
  _tile_release();
}

void gemv_ref(const int M, const int N, const BF16* A, const BF16* x, FP32* y) {
  int i, j;
  for (i = 0; i < M; i++) {
    FP32 sum = 0.0f;
    for (j = 0; j < N; j++) {
      sum += A[i * N + j] * x[j];
    }
    y[i] = sum;
  }
}