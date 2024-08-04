#include "amx.h"
#include "utils.cpp"
void tmul_int8_uu() {
  UINT8 src1[1024];
  UINT8 src2[1024];
  INT32 res[1024 / 4];
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
  init_buffer(src1, (UINT8)200, M, 4 * K);
  print_buffer(src1, M, 4 * K);

  init_buffer(src2, (UINT8)200, K, 4 * N);
  print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  _tile_dpbuud(0, 1, 2);

  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}
void tmul_int8_us() {
  UINT8 src1[1024];
  INT8 src2[1024];
  INT32 res[1024 / 4];
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
  init_buffer(src1, (UINT8)200, M, 4 * K);
  print_buffer(src1, M, 4 * K);

  init_buffer(src2, (INT8)-1, K, 4 * N);
  print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  _tile_dpbusd(0, 1, 2);

  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}
void tmul_int8_su() {
  INT8 src1[1024];
  UINT8 src2[1024];
  INT32 res[1024 / 4];
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
  init_buffer(src1, (INT8)-1, M, 4 * K);
  print_buffer(src1, M, 4 * K);

  init_buffer(src2, (UINT8)200, K, 4 * N);
  print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  _tile_dpbsud(0, 1, 2);

  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}
void tmul_int8_ss() {
  INT8 src1[1024];
  INT8 src2[1024];
  INT32 res[1024 / 4];
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
  init_buffer(src1, (INT8)-1, M, 4 * K);
  print_buffer(src1, M, 4 * K);

  init_buffer(src2, (INT8)-1, K, 4 * N);
  print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  _tile_dpbssd(0, 1, 2);

  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}