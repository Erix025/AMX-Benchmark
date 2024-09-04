#include <chrono>

#include "amx.h"
#include "utils.h"

void tmul_benchmark_int8_uu() {
  UINT8 src1[1024];
  UINT8 src2[1024];
  INT32 res[1024 / 4];
  int M, N, K;
  M = 16;
  K = 16;
  N = 16;

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
  //   print_buffer(src1, M, 4 * K);

  init_buffer(src2, (UINT8)200, K, 4 * N);
  //   print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_dpbuud(0, 1, 2);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  int ops_per_iter = M * N * K * 8;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n"
            << "Tile Multiply Benchmark (INT8_UU)\n";
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter * ops_per_iter / elapsed.count() /
                   1e12
            << " TOPS\n";
  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  //   print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void tmul_benchmark_int8_ss() {
  INT8 src1[1024];
  INT8 src2[1024];
  INT32 res[1024 / 4];
  int M, N, K;
  M = 16;
  K = 16;
  N = 16;

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
  init_buffer(src1, (INT8)200, M, 4 * K);
  //   print_buffer(src1, M, 4 * K);

  init_buffer(src2, (INT8)200, K, 4 * N);
  //   print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_dpbssd(0, 1, 2);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  int ops_per_iter = M * N * K * 8;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n"
            << "Tile Multiply Benchmark (INT8_SS)\n";
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter * ops_per_iter / elapsed.count() /
                   1e12
            << " TOPS\n";
  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  //   print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void tmul_benchmark_int8_us() {
  UINT8 src1[1024];
  INT8 src2[1024];
  INT32 res[1024 / 4];
  int M, N, K;
  M = 16;
  K = 16;
  N = 16;

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
  //   print_buffer(src1, M, 4 * K);

  init_buffer(src2, (INT8)200, K, 4 * N);
  //   print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_dpbusd(0, 1, 2);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  int ops_per_iter = M * N * K * 8;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n"
            << "Tile Multiply Benchmark (INT8_US)\n";
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter * ops_per_iter / elapsed.count() /
                   1e12
            << " TOPS\n";
  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  //   print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void tmul_benchmark_int8_su() {
  INT8 src1[1024];
  UINT8 src2[1024];
  INT32 res[1024 / 4];
  int M, N, K;
  M = 16;
  K = 16;
  N = 16;

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
  init_buffer(src1, (INT8)200, M, 4 * K);
  //   print_buffer(src1, M, 4 * K);

  init_buffer(src2, (UINT8)200, K, 4 * N);
  //   print_buffer(src2, K, 4 * N);

  // Init dst matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  // Compute dot-product of bytes in tiles
  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_dpbsud(0, 1, 2);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  int ops_per_iter = M * N * K * 8;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n"
            << "Tile Multiply Benchmark (INT8_SU)\n";
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter * ops_per_iter / elapsed.count() /
                   1e12
            << " TOPS\n";
  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  //   print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void tmul_benchmark_bf16(int M, int N, int K) {
  BF16 src1[1024 / 2];
  BF16 src2[1024 / 2];
  FP32 res[1024 / 4];

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
  init_buffer(src1, (BF16)1.1, M, 2 * K);
  // print_buffer(src1, M, 2 * K);

  init_buffer(src2, (BF16)1.1, K, 2 * N);
  // print_buffer(src2, K, 2 * N);

  // Init dst matrix buffers with data
  init_buffer(res, (FP32)1.0, M, N);

  // Load tile rows from memory
  _tile_loadd(1, src1, K * 4);
  _tile_loadd(2, src2, N * 4);
  _tile_loadd(0, res, N * 4);

  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_dpbf16ps(0, 1, 2);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  int ops_per_iter = M * N * K * 4;
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n"
            << "Tile Multiply Benchmark (BF16)\n"
            << "M: " << M << " N: " << N << " K: " << K << std::endl;
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter * ops_per_iter / elapsed.count() /
                   1e12
            << " TOPS\n";
  // Store the tile data to memory
  _tile_stored(0, res, N * 4);
  // print_buffer(res, M, N);

  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void tload_benchmark() {
  UINT8 src1[1024];
  UINT8 src2[1024];
  INT32 res[1024 / 4];
  int M, N, K;
  M = 16;
  K = 16;
  N = 16;

  // Request permission to linux kernel to run AMX
  if (!enable_amx()) exit(-1);

  __tilecfg config = {0};
  {
    config.palette_id = 1;
    config.start_row = 0;

    config.colsb[0] = N * 4;
    config.rows[0] = M;
  }
  _tile_loadconfig(&config);

  // Init src matrix buffers with data
  init_buffer(res, 1, M, N);

  // Load tile rows from memory
  // Compute dot-product of bytes in tiles
  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_loadd(0, res, N * 4);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n" << "Tile Load Benchmark\n";
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter / elapsed.count() / 1e9
            << " G iters/s |"
            << (double)test_frequency * iter * M * K * 4 / elapsed.count() / 1e9
            << " GB/s\n";
  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void tstore_benchmark() {
  UINT8 src1[1024];
  UINT8 src2[1024];
  INT32 res[1024 / 4];
  int M, N, K;
  M = 16;
  K = 16;
  N = 16;

  // Request permission to linux kernel to run AMX
  if (!enable_amx()) exit(-1);

  __tilecfg config = {0};
  {
    config.palette_id = 1;
    config.start_row = 0;

    config.colsb[0] = N * 4;
    config.rows[0] = M;
  }
  _tile_loadconfig(&config);

  // Init src matrix buffers with data
  init_buffer(res, 1, M, N);

  _tile_loadd(0, res, N * 4);
  // Load tile rows from memory
  // Compute dot-product of bytes in tiles
  int test_frequency = 1000;
  int iter = 1000000;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < test_frequency; i++) {
    for (int j = 0; j < iter; j++) {
      _tile_stored(0, res, N * 4);
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "====================\n" << "Tile Store Benchmark\n";
  std::cout << "Elapsed time: " << elapsed.count() << " s\n";
  std::cout << "Throughput: "
            << (double)test_frequency * iter / elapsed.count() / 1e9
            << " G iters/s |"
            << (double)test_frequency * iter * M * N * 4 / elapsed.count() / 1e9
            << " GB/s\n";
  // Release the tile configuration to return to the init state,
  // which releases all storage it currently holds
  _tile_release();
}

void benchmark_all() {
  tload_benchmark();
  tmul_benchmark_int8_uu();
  tmul_benchmark_int8_su();
  tmul_benchmark_int8_us();
  tmul_benchmark_int8_ss();
  tmul_benchmark_bf16(16, 16, 16);
  tstore_benchmark();
}

void benchmark_bf16_shapes() {
  for (int i = 1; i <= 16; i++) {
    for (int j = 1; j <= 16; j++) {
      for (int k = 1; k <= 16; k++) {
        tmul_benchmark_bf16(i, j, k);
      }
    }
  }
}