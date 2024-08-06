#pragma once
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>
#include <map>

#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18

using UINT8 = uint8_t;
using INT8 = int8_t;
using INT32 = int32_t;
using BF16 = __bf16;
using FP16 = _Float16;
using FP32 = float;

// Define tile config data structure
typedef struct __tile_config {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved_0[14];
  uint16_t colsb[16];
  uint8_t rows[16];
} __tilecfg;

bool enable_amx();