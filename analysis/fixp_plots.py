import matplotlib.pyplot as plt
import numpy as np
import torch as to
import math as ma

g = to.nn.GELU()
re = to.nn.ReLU()
unit = lambda x: x


def gelu(q):
    return (q * 0.5) * (1 + ma.erf(q / ma.sqrt(2)))


def relu(q):
    if q > 0:
        return q
    else:
        return 0


def gelu_mod(q):
    return gelu(q) - relu(q)


def quantize_7(x):
    return int(x * 128) / 128

markersize=0


htable = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
          '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
          'e': 14, 'f': 15}

f = np.exp


def hex2int(h, acc=0):
    if len(h) == 0:
        return acc
    else:
        return hex2int(h[1:], 16 * acc + htable[h[0]])


def unpack_float(q):
    sign = q < 0
    h = float(q).hex()
    dot = h.find('.')
    pspot = h.find('p')
    mantissa = h[dot + 1:min(dot + 3, pspot)]
    mantissa = (hex2int(mantissa) // 2) / 128
    exponent = int(h[h.find('p') + 1:])
    return sign, exponent, mantissa


def repack_float(s, e, m, verbose=False):
    s_str = "-" if s else ""
    q = hex(int(m * 256))[2:]
    if len(q) == 1:
        q = "0" + q
    m_str = "1." + q
    flt = f"{s_str}{m_str}p{int(e)}"
    # if verbose:
    #     print(flt)
    return float.fromhex(flt)


def x_map(x):
    s, e, m = unpack_float(x)
    if s:
        return e + m
    else:
        return e + m


def x_map_ar(x):
    return np.array([x_map(xp) for xp in x])


SIGN = lambda __x: __x
x_e_min = 1
x_e_max = 3

_, lim_y_e, lim_y_m = unpack_float(f(SIGN(2 ** x_e_min)))
d_lookup = {}
for x_ in range(x_e_min, x_e_max + 1):
    _, flo_e, flo_m = unpack_float(f(SIGN(2 ** x_)))
    _, fhi_e, fhi_m = unpack_float(f(SIGN(2 ** (x_ + 1))))
    d_lookup[x_] = fhi_e - flo_e + fhi_m - flo_m
    q = hex((int(flo_e) + 127) * 128 + int(flo_m * 128))[2:]
    print(f"  BASE[{x_+127}]= 16'h{q}")
    print(f"OFFSET[{x_+127}]={d_lookup[x_]*128}")


for x_ in range(x_e_min, x_e_max + 1):
    _, e, m = unpack_float(f(SIGN(2 ** x_)))
    q = hex((int(e) + 127) * 128 + int(m * 128))[2:]
    print(x_, ": ", e, m * 128, f"16'h{q}")

for key in d_lookup.keys():
    print(key, ": ", d_lookup[key] * 128)


def get_approx(x):
    _, e, m = unpack_float(x)
    if e < x_e_min:
        return False, 0, 0
    su = lim_y_m
    for i in range(x_e_min, e):
        su += d_lookup[i]
    su += d_lookup[e] * m
    assert (m < 1)
    e_app = np.floor(lim_y_e + su)
    m_app = int((su * 128) % 128)
    # print("APPROX: ", e_app, m_app)
    return False, e_app, m_app


def plot_discontinuous_f_on(x, fapprox, fgold):
    # ax.set_yticks(range(-50, 50))
    # ax2.set_yticks(np.arange(0, 1, 1.0 / 128), labels="")
    ax.grid(True)
    ax2.grid(True)
    e_pts = []
    x_pts = []
    m_pts = []
    x_approx_pts = []
    e_approx_pts = []
    m_approx_pts = []
    err_x_pts = []
    err_pts = []
    err_real_pts = []
    err = 0
    errl1 = 0
    biggest_diff = 0
    current_gold = None
    current_approx = None

    for x_i in x:
        x_i_s, x_i_e, x_i_m = unpack_float(x_i)
        x_i_rnd = repack_float(x_i_s, x_i_e, x_i_m)
        y_gold = fgold(x_i_rnd)
        _, e_approx, m_approx = fapprox(x_i_rnd)
        _, e_gold, m_gold = unpack_float(y_gold)
        y_approx = repack_float(False, e_approx, m_approx / 128, verbose=True)
        diff = np.abs((e_approx + m_approx / 128) - (e_gold + m_gold))

        errl1 += diff
        err += diff ** 2
        err_pts.append(diff * 128)
        err_real_pts.append(np.abs(y_approx - y_gold) / y_gold)
        err_x_pts.append(x_i)

        if diff > biggest_diff:
            print("biggest diff @ ", x_i, y_gold, y_approx)
            biggest_diff = diff
        xm = x_map(x_i)
        # print(x_i, y_gold, y_approx)
        if current_gold != e_gold:
            if current_gold is not None:
                ax.plot(x_pts, e_pts, marker='.', linewidth=1, markersize=markersize, color='black')
                ax2.plot(x_pts, m_pts, marker='.', linewidth=1, markersize=markersize, color='black')
            current_gold = e_gold
            for a in [e_pts, x_pts, m_pts]:
                a.clear()
        x_pts.append(x_i)
        e_pts.append(e_gold)
        m_pts.append(m_gold)

        if current_approx != e_approx:
            current_approx = e_approx
            for a in [e_approx_pts, x_approx_pts, m_approx_pts]:
                a.clear()
        x_approx_pts.append(xm)
        e_approx_pts.append(e_approx)
        m_approx_pts.append(m_approx / 128)
    ax.plot(x_pts, e_pts, marker='.', linewidth=1, markersize=markersize, color='black')
    ax2.plot(x_pts, m_pts, marker='.', linewidth=1, markersize=markersize, color='black')

    mv_x = []
    mv_real = []
    window_sz = 32
    for i in range(window_sz, len(err_real_pts) - window_sz):
        window = err_real_pts[i - window_sz:i + 1:window_sz]
        mv_real.append(sum(window) / len(window))
        mv_x.append(err_x_pts[i])


d__ = 1/128
x = np.arange(2**x_e_min, 2**x_e_max, d__)

to_sup_map = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
              "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
              "8": "\u2078", "9": "\u2079", "-": "\u207b"}


def to_sup(a: str):
    if a:
        return to_sup_map[a[0]] + to_sup(a[1:])
    else:
        return ""


fig, axs = plt.subplots(1, 2, figsize=(5, 3), dpi=800)
############### EXP + #####################
ax, ax2 = axs
ax.set_ylabel("OUTPUT EXPONENT")
ax2.set_ylabel("OUTPUT MANTISSA")
ax.set_xlabel("(a)")
ax2.set_xlabel("(b)")
plot_discontinuous_f_on(x, get_approx, f)
fig.tight_layout()
fig.savefig('fig1.pdf')