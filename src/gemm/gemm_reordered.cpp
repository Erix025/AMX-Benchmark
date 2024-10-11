#include "amx.h"
#include "utils.h"

namespace AMXBench {
namespace gemm {

void reordered(const int M, const int K, const int N, const BF16* A,
               const BF16* B, FP32* C) {
    // enable amx
    if (!enable_amx()) exit(-1);
    // init default config
    __tilecfg default_config = {0};
    {
        default_config.palette_id = 1;
        default_config.start_row = 0;
        // C00
        default_config.rows[0] = 16;
        default_config.colsb[0] = 64;
        // C01
        default_config.rows[1] = 16;
        default_config.colsb[1] = 64;
        // C10
        default_config.rows[2] = 16;
        default_config.colsb[2] = 64;
        // C11
        default_config.rows[3] = 16;
        default_config.colsb[3] = 64;
        // A0
        default_config.rows[4] = 16;
        default_config.colsb[4] = 64;
        // A1
        default_config.rows[5] = 16;
        default_config.colsb[5] = 64;
        // B0
        default_config.rows[6] = 16;
        default_config.colsb[6] = 64;
        // B1
        default_config.rows[7] = 16;
        default_config.colsb[7] = 64;
    }
    int m, n, k;
    const int tile_row = 16;
    const int tile_stride = 4 / sizeof(BF16);
    const int tile_col_bf16 = 64 / sizeof(BF16);
    const int tile_col_fp32 = 64 / sizeof(FP32);
    const int tile_size = tile_row * tile_col_bf16;
    const int group_size = 2;
    int reordered_N = N * 2;

    _tile_loadconfig(&default_config);
    for (int m = 0; m < M / tile_row / group_size; m++) {
        for (int n = 0; n < N / tile_col_fp32 / group_size; n++) {
            // load C
            _tile_loadd(0,
                        C + m * group_size * tile_row * N +
                            n * group_size * tile_col_fp32,
                        N * sizeof(FP32));
            _tile_loadd(1,
                        C + m * group_size * tile_row * N +
                            (n * group_size + 1) * tile_col_fp32,
                        N * sizeof(FP32));
            _tile_loadd(2,
                        C + (m * group_size + 1) * tile_row * N +
                            n * group_size * tile_col_fp32,
                        N * sizeof(FP32));
            _tile_loadd(3,
                        C + (m * group_size + 1) * tile_row * N +
                            (n * group_size + 1) * tile_col_fp32,
                        N * sizeof(FP32));
            BF16* A_tile = const_cast<BF16*>(A) + m * group_size * tile_row * K;
            BF16* B_tile = const_cast<BF16*>(B) + n * group_size * tile_row * K;
            for (int k = 0; k < K / tile_col_bf16; k++) {
                // load A
                _tile_loadd(4, A_tile + k * group_size * tile_size,
                            tile_col_bf16 * sizeof(BF16));
                _tile_loadd(5, A_tile + (k * group_size + 1) * tile_size,
                            tile_col_bf16 * sizeof(BF16));
                // load B
                _tile_loadd(6, B_tile + k * group_size * tile_size,
                            tile_col_bf16 * sizeof(BF16));
                _tile_loadd(7, B_tile + (k * group_size + 1) * tile_size,
                            tile_col_bf16 * sizeof(BF16));
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0,
                         C + m * group_size * tile_row * N +
                             n * group_size * tile_col_fp32,
                         N * sizeof(FP32));
            _tile_stored(1,
                         C + m * group_size * tile_row * N +
                             (n * group_size + 1) * tile_col_fp32,
                         N * sizeof(FP32));
            _tile_stored(2,
                         C + (m * group_size + 1) * tile_row * N +
                             n * group_size * tile_col_fp32,
                         N * sizeof(FP32));
            _tile_stored(3,
                         C + (m * group_size + 1) * tile_row * N +
                             (n * group_size + 1) * tile_col_fp32,
                         N * sizeof(FP32));
        }
    }
}

void pack_matrix_A(const int M, const int K, BF16* A) {
    BF16* A_copy = new BF16[M * K];
    std::memcpy(A_copy, A, M * K * sizeof(BF16));
    const int tile_row = 16;
    const int tile_col = 16 * (4 / sizeof(BF16));
    const int group_size = 2;  // 1 group = 2x1 tiles
    BF16* dst = A;
    for (int m = 0; m < M / tile_row / group_size; m++) {
        for (int k = 0; k < K / tile_col; k++) {
            for (int ii = 0; ii < group_size; ii++) {
                BF16* src = A_copy + (m * group_size + ii) * tile_row * K +
                            k * tile_col;
                for (int i = 0; i < tile_row; i++) {
                    for (int j = 0; j < tile_col; j++) {
                        *dst++ = *src++;
                    }
                    src += K - tile_col;
                }
            }
        }
    }
}

void pack_matrix_B(const int M, const int N, BF16* B) {
    BF16* B_copy = new BF16[M * N];
    std::memcpy(B_copy, B, M * N * sizeof(BF16));
    const int tile_size = 16;
    const int tile_row = tile_size;
    const int tile_col = tile_size * (4 / sizeof(BF16));
    const int group_size = 2;  // 1 group = 1x2 tiles
    BF16* dst = B;
    for (int n = 0; n < N / tile_row / group_size; n++) {
        for (int m = 0; m < M / tile_col; m++) {
            for (int ii = 0; ii < group_size; ii++) {
                BF16* src = B_copy + m * tile_col * N +
                            (n * group_size + ii) * tile_row;
                BF16* src_row1 = src;
                BF16* src_row2 = src + N;
                for (int i = 0; i < tile_row; i++) {
                    for (int j = 0; j < tile_size; j++) {
                        *dst++ = *src_row1++;
                        *dst++ = *src_row2++;
                    }
                    src_row1 += 2 * N - tile_size;
                    src_row2 += 2 * N - tile_size;
                }
            }
        }
    }
}

namespace test {
void test_pack_matrix_A() {
    const int M = 64;
    const int K = 64;
    BF16* A = new BF16[M * K];
    range_buffer(A, M, K);
    print_buffer(A, M, K);
    pack_matrix_A(M, K, A);
    print_buffer(A, M, K);
}

void test_pack_matrix_B() {
    const int M = 64;
    const int N = 64;
    BF16* B = new BF16[M * N];
    range_buffer(B, M, N);
    print_buffer(B, M, N);
    pack_matrix_B(M, N, B);
    print_buffer(B, M / 2, N * 2);
}
}  // namespace test

}  // namespace gemm
}  // namespace AMXBench