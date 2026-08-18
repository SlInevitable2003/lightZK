#pragma once
#include <cstddef>

inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }
inline size_t round_up(size_t a, size_t b) { return ceil_div(a, b) * b; }
inline size_t round_down(size_t a, size_t b) { return a / b * b; }
inline size_t log2_floor(size_t a) { return 63 - __builtin_clzll(a); }
