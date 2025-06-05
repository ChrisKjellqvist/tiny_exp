#include "tiny_exp.h"
#include <iostream>
#include <chrono>
#include <cmath>

using namespace fpE8M23;
const uint32_t c = 1'000'000;
const uint32_t MASK = 0xF;
int main() {
    float s = 0;
    for (int i = 0; i < c; ++i) {
        s += tiny_exp<float>(i & MASK);
    }

    float q = 0;
    auto s2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < c; ++i) {
        q += tiny_exp<float>(i & MASK);
    }
    auto e2 = std::chrono::high_resolution_clock::now();

    float r = 0;
    auto s1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < c; ++i) {
        r += std::exp<float>(i & MASK);
    }
    auto e1 = std::chrono::high_resolution_clock::now();

    __fp16 t = 0;
    auto s3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < c; ++i) {
        t += std::exp<__fp16>(i & MASK);
    }
    auto e3 = std::chrono::high_resolution_clock::now();

    printf("%f\n", q);
    printf("%f\n", r);
    printf("%f\n", s);
    printf("%f\n", (float)t);
    uint64_t fp32_us = std::chrono::duration_cast<std::chrono::microseconds>(e1-s1).count();
    uint64_t tiny_us = std::chrono::duration_cast<std::chrono::microseconds>(e2-s2).count();
    uint64_t fp16_us = std::chrono::duration_cast<std::chrono::microseconds>(e3-s3).count();
    std::cout << "Regular FP32 Exp took " << fp32_us << "µs" << std::endl;
    std::cout << "Regular FP16 Exp took " << fp16_us << "µs" << std::endl;
    std::cout << "Tiny FP32 Exp took " << tiny_us << "µs" << std::endl;
    printf("Tiny is %0.2fX faster than FP32\n", double(fp32_us)/double(tiny_us));
    printf("Tiny op takes %0.2fns\n", double(fp32_us)/c*1'000);
}