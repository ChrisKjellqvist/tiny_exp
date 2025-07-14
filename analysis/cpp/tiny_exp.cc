#include "tiny_exp.h"
#include <bit>

namespace fpE8M7 {
template<>
uint16_t tiny_exp<uint16_t>(const uint16_t &as_bit) {
    uint16_t S = as_bit >> 15;
    uint16_t E = (as_bit >> 7) & 0xff;
    if (E >= 133) [[unlikely]] {
        if (S) return 0;
        else return 0x7f80;
    }
    if (E < 120) [[unlikely]] {
        return 0x3f80;
    }
    uint16_t Enorm = E - 120;
    uint32_t fused = lookup[S][Enorm];
    uint32_t D = fused >> 16;
    uint16_t base = fused & 0xffff;
    uint32_t M = (as_bit & 0x7f) * D;
    return base + (1 - (S << 1)) * static_cast<uint16_t>(M >> 7);
}
template<>
float tiny_exp<float>(const float &f) {
    uint32_t q = std::bit_cast<uint32_t>(f);
    uint16_t r = tiny_exp(static_cast<uint16_t>(q >> 16));
    return std::bit_cast<float>(static_cast<uint32_t>(r)<<16);
}
}
namespace fpE8M23 {
template<>
uint32_t tiny_exp<uint32_t>(const uint32_t &as_bit) {
    uint32_t S = as_bit >> 31;
    uint32_t E = (as_bit >> 23) & 0xff;
    if (E >= 133) [[unlikely]] {
        if (S) return 0;
        else return 0x7f800000;
    }
    if (E < 120) [[unlikely]] {
        return 0x3f800000;
    }
    uint32_t Enorm = E - 120;
    uint64_t fused = lookup[S][Enorm];
    uint64_t D = fused >> 32;
    uint32_t base = fused & 0xffffffff;
    uint64_t M = (as_bit & 0x7fffff) * D;
    return base + (1 - (S << 1)) * static_cast<uint32_t>(M >> 23);
}
template<>
float tiny_exp<float>(const float &f) {
    uint32_t q = std::bit_cast<uint32_t>(f);
    uint32_t r = tiny_exp(static_cast<uint32_t>(q >> 0));
    return std::bit_cast<float>(static_cast<uint32_t>(r)<<0);
}
}
namespace fpE5M10 {
template<>
uint16_t tiny_exp<uint16_t>(const uint16_t &as_bit) {
    uint16_t S = as_bit >> 15;
    uint16_t E = (as_bit >> 10) & 0x1f;
    if (E >= 18) [[unlikely]] {
        if (S) return 0;
        else return 0x7c00;
    }
    if (E < 11) [[unlikely]] {
        return 0x3c00;
    }
    uint16_t Enorm = E - 11;
    uint32_t fused = lookup[S][Enorm];
    uint32_t D = fused >> 16;
    uint16_t base = fused & 0xffff;
    uint32_t M = (as_bit & 0x3ff) * D;
    return base + (1 - (S << 1)) * static_cast<uint16_t>(M >> 10);
}
template<>
float tiny_exp<float>(const float &f) {
    uint32_t q = std::bit_cast<uint32_t>(f);
    uint16_t r = tiny_exp(static_cast<uint16_t>(q >> 16));
    return std::bit_cast<float>(static_cast<uint32_t>(r)<<16);
}
}