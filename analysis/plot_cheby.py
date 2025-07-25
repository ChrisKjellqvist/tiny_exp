import matplotlib.pyplot as plt
import numpy as np


polynomials = [
    lambda _: 1,
    lambda x: x
]
polynomials.append(lambda x: 2 * x * polynomials[1](x) - polynomials[0](x))
polynomials.append(lambda x: 2 * x * polynomials[2](x) - polynomials[1](x))
polynomials.append(lambda x: 2 * x * polynomials[3](x) - polynomials[2](x))

fig,ax = plt.subplots(figsize=(4, 2))

for i, poly in enumerate(polynomials):
    x = np.arange(-1, 1, 0.001)
    y = []
    for x_ in x:
        y.append(poly(x_))
    ax.plot(x, y, label=f"T_{i}")
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
fig.tight_layout()
fig.savefig(f"chebyshev.pdf")