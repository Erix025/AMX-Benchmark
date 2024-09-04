# vtune -collect performance-snapshot -result-dir ./vtune ./build/AMX-Benchmark
# rm -rf ./vtune-ht
# vtune -collect hotspots -result-dir ./vtune-ht ./build/AMX-Benchmark

rm -rf ./vtune-mem
vtune -collect memory-access -result-dir ./vtune-mem ./build/AMX-Benchmark