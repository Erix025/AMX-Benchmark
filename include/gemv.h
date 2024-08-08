#include "amx.h"
void gemv();

void gemv_tile(BF16* A, BF16* x, FP32* y);

void gemv_naive(const int M, const int N, const BF16* A, const BF16* x,
                FP32* y);

void gemv_ref(const int M, const int N, const BF16* A, const BF16* x, FP32* y);