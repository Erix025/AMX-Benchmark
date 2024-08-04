#include "amx.h"
bool enable_amx() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    std::cerr << "Failed to enable AMX" << std::endl;
    return false;
  }
  return true;
}
