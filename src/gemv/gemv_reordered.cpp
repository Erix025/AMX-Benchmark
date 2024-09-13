#include "amx.h"
#include "utils.h"
/**
 * GEMV optimized with 8 tile registers and reorder matrix.
 * Please make sure that the shape of A and x is supported.
 */
void gemv_reordered(const int M, const int N, const BF16* A, const BF16* x,
                    FP32* y) {
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
    default_config.rows[1] = 16;
    default_config.colsb[1] = 4;
    // A
    default_config.rows[2] = 16;
    default_config.colsb[2] = 64;
    default_config.rows[3] = 16;
    default_config.colsb[3] = 64;
    default_config.rows[4] = 16;
    default_config.colsb[4] = 64;
    default_config.rows[5] = 16;
    default_config.colsb[5] = 64;
    // x
    default_config.rows[6] = 16;
    default_config.colsb[6] = 4;
    default_config.rows[7] = 16;
    default_config.colsb[7] = 4;
  }
  int i, j;
  int tile_size = 16 * 32;

  BF16* A_pointer = const_cast<BF16*>(A);
  _tile_loadconfig(&default_config);
  for (i = 0; i <= M - 16 * 2; i += 16 * 2) {
    _tile_loadd(0, y + i, sizeof(FP32));
    _tile_loadd(1, y + i + 16, sizeof(FP32));
    for (j = 0; j <= N - 32 * 2; j += 32 * 2) {
      _tile_loadd(2, A_pointer + tile_size * 0, sizeof(BF16) * 32);
      _tile_loadd(3, A_pointer + tile_size * 1, sizeof(BF16) * 32);
      _tile_loadd(4, A_pointer + tile_size * 2, sizeof(BF16) * 32);
      _tile_loadd(5, A_pointer + tile_size * 3, sizeof(BF16) * 32);
      _tile_loadd(6, x + j, 2 * sizeof(BF16));
      _tile_loadd(7, x + j + 32, 2 * sizeof(BF16));
      _tile_dpbf16ps(0, 2, 6);
      _tile_dpbf16ps(0, 3, 7);
      _tile_dpbf16ps(1, 4, 6);
      _tile_dpbf16ps(1, 5, 7);
      A_pointer += tile_size * 4;
    }
    _tile_stored(0, y + i, sizeof(FP32));
    _tile_stored(1, y + i + 16, sizeof(FP32));
  }
}