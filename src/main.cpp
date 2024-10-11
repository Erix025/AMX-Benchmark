#include <chrono>

#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemm.h"
#include "gemv.h"
#include "sched.h"
#include "test.h"
#include "utils.h"
int main() {
    int M = 16384, N = 16384, K = 16384;
    gemm::benchmark(M, K, N, 1);
    // gemm::test::test_pack_matrix_A();
    // gemm::test::test_pack_matrix_B();
    return 0;
}
