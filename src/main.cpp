#include <chrono>

#include "amx.h"
#include "benchmark.h"
#include "examples.h"
#include "gemv.h"
#include "sched.h"
#include "utils.cpp"
int main() {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);    // 清除CPU集合
  CPU_SET(0, &cpuset);  // 将CPU核心0加入集合

  // 获取当前线程的ID
  pid_t pid = getpid();

  // 绑定线程到CPU核心0
  if (sched_setaffinity(pid, sizeof(cpuset), &cpuset) != 0) {
    perror("sched_setaffinity");
    return 1;
  }

  std::cout << "Thread is bound to CPU core 0" << std::endl;
  benchmark_all();
  // benchmark_gemv();
  return 0;
}
