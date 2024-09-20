#include "amx.h"
#include "utils.h"

namespace gemm {

void baseline(const int M, const int K, const int N, const BF16* A,
              const BF16* B, FP32* C) {
  // enable amx
  if (!enable_amx()) exit(-1);
  // init default config
  __tilecfg default_config = {0};
  {
    default_config.palette_id = 1;
    default_config.start_row = 0;
    // C
    default_config.rows[0] = 16;
    default_config.colsb[0] = 64;
    // A
    default_config.rows[1] = 16;
    default_config.colsb[1] = 64;
    // B
    default_config.rows[2] = 16;
    default_config.colsb[2] = 64;
  }
  int m, n, k;
  const int tile_row = 16;
  const int tile_stride = 4 / sizeof(BF16);
  const int tile_col_bf16 = 64 / sizeof(BF16);
  const int tile_col_fp32 = 64 / sizeof(FP32);

  int reordered_K = K / 2;
  int reordered_N = N * 2;

  _tile_loadconfig(&default_config);
  for (m = 0; m < M / 16; m++) {
    for (n = 0; n < reordered_N / 32; n++) {
      _tile_loadd(0, C + m * tile_row * N + n * tile_col_fp32,
                  sizeof(FP32) * N);
      for (k = 0; k < K / 32; k++) {
        _tile_loadd(1, A + m * tile_row * K + k * tile_col_bf16,
                    sizeof(BF16) * K);
        _tile_loadd(2, B + k * tile_row * reordered_N + n * tile_col_bf16,
                    sizeof(BF16) * reordered_N);
        _tile_dpbf16ps(0, 1, 2);
      }
      _tile_stored(0, C + m * tile_row * N + n * tile_col_fp32,
                   sizeof(FP32) * N);
    }
  }
}
}  // namespace gemm