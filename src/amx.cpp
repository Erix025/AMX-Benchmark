#include "amx.h"
bool enable_amx() {
  if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
    std::cerr << "Failed to enable AMX" << std::endl;
    return false;
  }
  return true;
}

void print_config() {
  __tilecfg temp;
  _tile_storeconfig(&temp);
  std::cout << "====== Tile Config ======" << std::endl;
  std::cout << "palette id: " << +temp.palette_id << std::endl;
  std::cout << "start row: " << +temp.start_row << std::endl;
  for (int i = 0; i < 8; i++) {
    std::cout << "[Tile " << i << "] colsb: " << temp.colsb[i]
              << " rows: " << +temp.rows[i] << std::endl;
  }
}