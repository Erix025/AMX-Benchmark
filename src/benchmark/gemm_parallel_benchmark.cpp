#include "amx.h"
#include "benchmark.h"
#include "gemm.h"
#include "utils.h"

namespace AMXBench {
namespace gemm {
void bench_gemm_parallel(int M, int K, int N, int max_iter, int thread_row,
                         int thread_col) {
    // benchmark for naive version
    BF16* A = new BF16[M * K];
    BF16* B = new BF16[K * N];
    FP32* C = new FP32[M * N];

    init_bench(M, K, N, A, B, C);

    bind_core(thread_row * thread_col);

    long long ops_per_iter = (long long)M * 2 * N * K;
    std::chrono::microseconds duration;

    pack_matrix_A(M, K, A);
    pack_matrix_B(K, N, B);

    duration = measure_time_with_preprocess<std::chrono::microseconds>(
        max_iter, gemm::parallel, init_bench_parallel, M, K, N, A, B, C,
        thread_row, thread_col);
    print_result("Parallel version", max_iter, duration, ops_per_iter);

    delete[] A;
    delete[] B;
    delete[] C;
}
}  // namespace gemm
}  // namespace AMXBench