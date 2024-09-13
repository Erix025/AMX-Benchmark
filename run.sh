# rm -rf ./vtune && vtune -collect performance-snapshot -result-dir ./vtune ./build/AMX-Benchmark > vtune.log
# rm -rf ./vtune-ht && vtune -collect hotspots -result-dir ./vtune-ht ./build/AMX-Benchmark

# rm -rf ./vtune-mem && vtune -collect memory-access -result-dir ./vtune-mem-reordered ./build/AMX-Benchmark > vtune-mem.log

rm -rf ./vtune-uarch && vtune -collect uarch-exploration -result-dir ./vtune-uarch ./build/AMX-Benchmark > vtune-uarch.log