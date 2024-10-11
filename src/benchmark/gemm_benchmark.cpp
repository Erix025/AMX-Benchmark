#include <getopt.h>

#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "benchmark.h"

void parse_args(int argc, char* argv[], int& M, int& N, int& K, int& max_iter,
                std::string& benchmark_name, int& thread_x, int& thread_y) {
    int opt;
    const char* const short_opts = "m:n:k:i:b:x:y:";
    const struct option long_opts[] = {
        {"m", required_argument, nullptr, 'm'},
        {"n", required_argument, nullptr, 'n'},
        {"k", required_argument, nullptr, 'k'},
        {"iter", required_argument, nullptr, 'i'},
        {"benchmark", required_argument, nullptr, 'b'},
        {"thread_x", required_argument, nullptr, 'x'},
        {"thread_y", required_argument, nullptr, 'y'},
        {nullptr, no_argument, nullptr, 0}};

    while ((opt = getopt_long(argc, argv, short_opts, long_opts, nullptr)) !=
           -1) {
        switch (opt) {
            case 'm':
                M = std::stoi(optarg);
                break;
            case 'n':
                N = std::stoi(optarg);
                break;
            case 'k':
                K = std::stoi(optarg);
                break;
            case 'i':
                max_iter = std::stoi(optarg);
                break;
            case 'b':
                benchmark_name = optarg;
                break;
            case 'x':
                thread_x = std::stoi(optarg);
                break;
            case 'y':
                thread_y = std::stoi(optarg);
                break;
            default:
                std::cerr << "Usage: " << argv[0]
                          << " [-m M] [-n N] [-k K] [-i max_iter] [-b "
                             "benchmark_name] [-x thread_x] [-y thread_y]"
                          << std::endl;
                exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024, max_iter = 10;
    std::string benchmark_name = "all";
    int thread_x = 1, thread_y = 1;

    parse_args(argc, argv, M, N, K, max_iter, benchmark_name, thread_x,
               thread_y);

    if (benchmark_name == "all") {
        AMXBench::gemm::bench_gemm_baseline(M, K, N, max_iter);
        AMXBench::gemm::bench_gemm_reordered(M, K, N, max_iter);
        AMXBench::gemm::bench_gemm_parallel(M, K, N, max_iter, thread_x,
                                            thread_y);
    } else if (benchmark_name == "baseline") {
        AMXBench::gemm::bench_gemm_baseline(M, K, N, max_iter);
    } else if (benchmark_name == "reordered") {
        AMXBench::gemm::bench_gemm_reordered(M, K, N, max_iter);
    } else if (benchmark_name == "parallel") {
        AMXBench::gemm::bench_gemm_parallel(M, K, N, max_iter, thread_x,
                                            thread_y);
    } else {
        std::cerr << "Unknown benchmark name: " << benchmark_name << std::endl;
        exit(EXIT_FAILURE);
    }
}