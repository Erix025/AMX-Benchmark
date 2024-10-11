#include <thread>
#include <vector>

#include "amx.h"
#include "gemm.h"
#include "utils.h"

namespace AMXBench {
namespace gemm {

void _kernel(const int M_thread, const int K, const int N_thread, const int N,
             const BF16* A, const BF16* B, FP32* C) {
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
    int reordered_N = N_thread * 2;

    _tile_loadconfig(&default_config);
    for (int m = 0; m < M_thread / tile_row / group_size; m++) {
        for (int n = 0; n < N_thread / tile_col_fp32 / group_size; n++) {
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

void parallel(const int M, const int K, const int N, const BF16* A,
              const BF16* B, FP32* C, const int thread_row,
              const int thread_col) {
    int M_per_thread = M / thread_row;
    int N_per_thread = N / thread_col;
    int num_threads = thread_row * thread_col;
    int group_size = 2;
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; i++) {
        int thread_x = i / thread_col;
        int thread_y = i % thread_col;
        const BF16* local_A = A + thread_x * M_per_thread * K;
        const BF16* local_B = B + thread_y * N_per_thread * K;
        FP32* local_C =
            C + thread_x * M_per_thread * N + thread_y * N_per_thread;
        if (i == num_threads - 1) {
            _kernel(M_per_thread, K, N_per_thread, N, local_A, local_B,
                    local_C);
            continue;
        }
        auto thread = std::thread(_kernel, M_per_thread, K, N_per_thread, N,
                                  local_A, local_B, local_C);
        threads.push_back(std::move(thread));
    }
    // wait for all threads to finish
    for (auto& thread : threads) {
        thread.join();
    }
}
}  // namespace gemm
}  // namespace AMXBench