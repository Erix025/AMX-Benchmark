#include <cstdint>

template <typename T>
void init_buffer(T *buf, T value, int32_t rows, int32_t cols);

template <typename T>
void print_buffer(T *buf, int32_t rows, int32_t cols);