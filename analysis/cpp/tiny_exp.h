#ifndef TINY_EXP_H
#define TINY_EXP_H
#include <cinttypes>

namespace fpE8M7 {
template <typename t>
t tiny_exp(const t &f);
const uint32_t lookup[2][14] = {
	{0x13f81, 0x23f82, 0x43f84, 0x93f88, 0x133f91, 0x2f3fa4, 0x5a3fd3, 0xbf402d, 0x16e40ec, 0x2e0425a, 0x5cd453a, 0xb884b07, 0x1712568f},
	{0x23f7e, 0x43f7c, 0x83f78, 0xf3f70, 0x1a3f61, 0x2c3f47, 0x5f3f1b, 0xb23ebc, 0x1743e0a, 0x2e73c96, 0x5be39af, 0xb8d33f1, 0x17192864}};
}
namespace fpE8M23 {
template <typename t>
t tiny_exp(const t &f);
const uint64_t lookup[2][14] = {
	{0x103053f810100, 0x20c263f820405, 0x4312f3f84102b, 0x8c9a83f88415a, 0x134fef3f910b02, 0x2eae5b3fa45af1, 0x5aef083fd3094c, 0xbe7ad1402df854, 0x16df15c40ec7325, 0x2dfead2425a6481, 0x5cd480b453a4f53, 0xb880aa04b07975e, 0x17118ac3568fa1fe},
	{0x1fa093f7e01fe, 0x3e84a3f7c07f5, 0x7a24c3f781fab, 0xe920e3f707d5f, 0x1a8bd53f61eb51, 0x2c19e53f475f7c, 0x5eeae63f1b4597, 0xb1c55c3ebc5ab1, 0x1748aa83e0a9555, 0x2e629a53c960aad, 0x5be362b39afe108, 0xb8d87b633f1aadd, 0x1718d48428642327}};
}
namespace fpE5M10 {
template <typename t>
t tiny_exp(const t &f);
const uint32_t lookup[2][8] = {
	{0x43c04, 0xa3c08, 0x173c12, 0x3ed3c29, 0x4204016, 0xbf74436, 0x17f0502d},
	{0x83838, 0xd3830, 0x163823, 0x3ef380d, 0x419341e, 0xbfa3005, 0x17f4240b}};
}
#endif