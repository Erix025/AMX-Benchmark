#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "amx.h"
#include "gemv.h"
#include "utils.h"
void _gemv_tile(BF16* A, BF16* x, FP32* y) {
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

void example_gemv_tile() {
  BF16 A[16 * 32];
  BF16 x[32];
  FP32 y[16];
  // init data
  init_buffer(A, (BF16)0.1, 16, 32);
  init_buffer(x, (BF16)1, 16, 2);
  init_buffer(y, (FP32)0, 16, 1);

  _gemv_tile(A, x, y);

  print_buffer(y, 16, 1);
}

void example_gemv(int M, int N) {
  BF16* A = new BF16[M * N];
  BF16* x = new BF16[N];
  FP32* y = new FP32[M];
  random_buffer(A, M, N);
  random_buffer(x, N, 1);
  init_buffer(y, (FP32)0, M, 1);
  gemv_reordered(M, N, A, x, y);
}
