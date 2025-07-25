#include <cmath>
#include <cstdio>

inline double fast_exp(double y) {
    double d;
    *(reinterpret_cast<int*>(&d) + 0) = 0;
    *(reinterpret_cast<int*>(&d) + 1) = static_cast<int>(1512775 * y + 1072632447);
    return d;
}

int main() {
	for (int i = 0; i < 10; ++i) {
		double delta = 1.0 / (1 << i);
		for (int j = 0; j < 10; ++j) {
			double n = -(j + delta * 4);
			auto gold = expf(n);
			auto approx = fast_exp(n);
			// printf("[%f] err: %0.4f\n", n, abs(gold - approx)/gold*100);
		}
	}
	constexpr float a = (1 << 23) / 0.69314718f;
    constexpr float b = (1 << 23) * (127 - 0.043677448f);
	uint32_t ai = reinterpret_cast<const int&>(a);
	uint32_t bi = reinterpret_cast<const int&>(b);
	printf("%08x %08x\n", ai, bi);
	
}