#include "amx.h"
#include "utils.h"
void tmul_complex() {
  COMPLEX src1[1024 / 4];
  COMPLEX src2[1024 / 4];
  FP32 res[1024 / 4];
  int M, N, K;
  M = 4;
  K = 4;
  N = 4;

  // Request permission to linux kernel to run AMX
  if (!enable_amx()) exit(-1);

  __tilecfg config = {0};
  {
    config.palette_id = 1;
    config.start_row = 0;

    config.colsb[0] = N * 4;
    config.rows[0] = M;

    config.colsb[1] = 4 * K;
    config.rows[1] = M;

    config.colsb[2] = 4 * N;
    config.rows[2] = K;
  }
  _tile_loadconfig(&config);

  // Init src matrix buffers with data
  init_buffer(src1, (COMPLEX){1, 1}, M, K);
  print_buffer(src1, M, K);

  init_buffer(src2, (COMPLEX){1, 1}, K, N);
  print_buffer(src2, K, N);

  // Init dst matrix buffers with data
  init_buffer(res, (FP32)1.0, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  // _tile_cmmrlfp16ps(0, 1, 2);

  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}