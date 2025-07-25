import numpy as np
import matplotlib.pyplot as plt

f = np.exp
d_size = 1
n_deg = 3

polynomials = []

mantissa_used_as_index = 0
mantissa_wid = 7
mantissa_tot = 1 << mantissa_wid
e_wid: int = 8

htable = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "a": 10,
    "b": 11,
    "c": 12,
    "d": 13,
    "e": 14,
    "f": 15,
}


def hex2int(h, acc=0):
    if len(h) == 0:
        return acc
    else:
        return hex2int(h[1:], 16 * acc + htable[h[0]])


shamt = 0 if mantissa_wid % 4 == 0 else 4 - (mantissa_wid % 4)
if mantissa_wid % 4 == 0:
    hex_digs = mantissa_wid
else:
    hex_digs = (mantissa_wid + shamt) // 4


def unpack_float(q, verbose=False):
    sign = q < 0
    h = float(q).hex()
    dot = h.find(".")
    pspot = h.find("p")
    mantissa = h[dot + 1 : min(dot + hex_digs + 1, pspot)]
    if verbose:
        print(hex_digs, " ", mantissa)
    inty = hex2int(mantissa)
    mantissa = (inty // (1 << shamt)) / mantissa_tot
    exponent = int(h[h.find("p") + 1 :])
    if verbose:
        print(f"{q} -> {h}, {inty}, {(sign,exponent,mantissa)}")
    return sign, exponent, mantissa


for d in range(-64, 64, d_size):
    domain = [d, d + d_size]
    initialization_points = np.arange(
        domain[0], domain[1], (domain[1] - domain[0]) / (n_deg + 1)
    ).tolist() + [domain[1]]
    assert len(initialization_points) == n_deg + 2
    elim_matrix = np.ndarray((n_deg + 2, n_deg + 2))
    for i in range(n_deg + 2):
        elim_matrix[i, -1] = (-1) ** i
        for j in range(n_deg + 1):
            elim_matrix[i, j] = initialization_points[i] ** j
    f_ar = [f(x_) for x_ in initialization_points]

    for i in range(n_deg + 2):
        f_ar[i] /= elim_matrix[i, i]
        elim_matrix[i, :] /= elim_matrix[i, i]
        for j in range(n_deg + 2):
            if j == i:
                continue
            m = elim_matrix[j, i]
            elim_matrix[j, :] -= elim_matrix[i, :] * m
            f_ar[j] -= m * f_ar[i]
    polynomials.append(f_ar.copy())

fig, ax = plt.subplots()


def to_flt_hex_str(s, e, m):
    flt_s: int = (1 if s else 0) << (e_wid + mantissa_wid)
    flt_adjusted_e: int = int(e) + ((1 << (e_wid - 1)) - 1)
    flt_adjusted_e_shift: int = flt_adjusted_e << mantissa_wid
    flt_adjusted_m: int = int(m * mantissa_tot)
    flt_combined_raw: int = flt_s | flt_adjusted_e_shift | flt_adjusted_m
    return hex(flt_combined_raw)[2:]

print(f"wire [15:0] constants [0:{n_deg}][0:{len(polynomials)-1}];")
for n in range(n_deg + 1):
    for i in range(len(polynomials)):
        s, e, m = unpack_float(polynomials[i][n])
        print(f"assign constants[{n}][{i}] = 16'h{to_flt_hex_str(s, e, m)};")


def eval_f(x):
    s = 0
    for i in range(len(f_ar) - 1):
        s += f_ar[i] * (x**i)
    return s


x = np.arange(domain[0] - 1, domain[1] + 1, 0.01)
approx = [eval_f(x_) for x_ in x]
ax.plot(x, approx, label="approx", linewidth=4, color="blue")
ax.plot(x, [f(x_) for x_ in x], label="gold", color="red")
ax.plot(
    [domain[0], domain[0]],
    [min(approx), max(approx)],
    color="black",
    linestyle="dashed",
)
ax.plot(
    [domain[1], domain[1]],
    [min(approx), max(approx)],
    color="black",
    linestyle="dashed",
)
# print(approx)
fig.legend()
fig.tight_layout()
fig.savefig("remez.pdf")
