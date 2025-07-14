import matplotlib.pyplot as plt
import numpy as np
import torch as to
import math as ma

def quantize_7(x):
    return int(x * 128) / 128


htable = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
          '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
          'e': 14, 'f': 15}

def exp(x):
    return np.exp(x)


def gen_htan(x, L, k, c, z):
    return L / (1 + np.exp(-k * (x - c))) - z


def silu(x):
    return gen_htan(x, L=1, k=1, c=0, z=0)

def htan(x):
    return gen_htan(x, L=2, k=2, c=0, z=1)

def sin(x):
    return np.exp(-(x**2))

# proposed uarch:
#   FLOAT( LUT_base[E] + M * LUT_off[E] ) + MUX(?, REG, 0)


f = exp
SIGN = lambda __x: __x

x_e_min = 1
x_e_max = 4
nmant = 4

plot_err = False
plot_approx = True
if plot_approx:
    dot_size = 4
else:
    dot_size = 0



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


_, lim_y_e, lim_y_m = unpack_float(f(SIGN(2 ** x_e_min)))
d_lookup = {}
for x_ in np.arange(x_e_min, x_e_max + 1, 1.0/(1<<nmant)):
    print(x_)
    intpart = int(x_)
    fracpart = x_-intpart
    start = SIGN(2 ** intpart)*(1+fracpart)
    goal = SIGN(2 ** intpart)*(1+fracpart+1.0/(1<<nmant))
    _, flo_e, flo_m = unpack_float(f(start))
    _, fhi_e, fhi_m = unpack_float(f(goal))
    # q = hex((int(flo_e) + 127) * 128 + int(flo_m * 128))[2:]
    comb = intpart + fracpart
    slope = (fhi_e - flo_e + fhi_m - flo_m) * (1 << nmant)
    d_lookup[comb] = ((flo_e, flo_m*128), slope)
    # print(f"  BASE[{x_+127}]= 16'h{q}")
    # print(f"OFFSET[{x_+127}]={d_lookup[comb][1]*128}")

for k in d_lookup.keys():
    b, c = d_lookup[k]
    print(" (", k, ") : ", b, " ", c, " ", int(c * 128))

def get_approx(x):
    _, e, m = unpack_float(x)
    frac = ma.floor(m * (1 << nmant)) / (1 << nmant)
    k = e + frac 
    g = d_lookup[k]

    e_app, m_app = g[0]
    mmult = (m * 128) % (128 >> nmant)
    m_app += mmult * g[1]
    m_app = int(m_app)

    e_app += m_app // 128
    m_app %= 128
    # print("APPROX: ", e_app, m_app)
    return False, e_app, m_app


def plot_discontinuous_f_on(x, fapprox, fgold):
    xticks = [f"2{to_sup(str(i))}" for i in range(x_e_min, x_e_max + 2)]
    ax.set_xticks(range(x_e_min, x_e_max + 2), labels=xticks)
    ax2.set_xticks(range(x_e_min, x_e_max + 2), labels=xticks)
    # ax.set_yticks(range(-50, 50))
    # ax2.set_yticks(np.arange(0, 1, 1.0 / 128), labels="")
    ax.grid(True)
    ax2.grid(True)
    ax.set_ylabel("Output Exponent")
    ax2.set_ylabel("Output Mantissa")
    if plot_err:
        ax3.grid(True)
        ax4.grid(True)
        ax3.set_ylabel("Error (distance)")
        ax4.set_ylabel("Error (%)")
    current_gold = None
    current_approx = None
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
    realerr = 0
    biggest_diff = 0
    has_done_once = False

    for x_i in x:
        x_i_s, x_i_e, x_i_m = unpack_float(x_i)
        x_i_rnd = repack_float(x_i_s, x_i_e, x_i_m)
        y_gold = fgold(x_i_rnd)
        _, e_approx, m_approx = fapprox(x_i_rnd)
        s, e_gold, m_gold = unpack_float(y_gold)
        y_approx = repack_float(s, e_approx, m_approx / 128, verbose=True)
        diff = np.abs((e_approx + m_approx / 128) - (e_gold + m_gold))

        errl1 += diff
        err += diff ** 2
        err_pts.append(diff * 128)
        err_real_pts.append(np.abs(y_approx - y_gold) / y_gold * 100)
        err_x_pts.append(x_i)
        realerr += abs(y_gold - y_approx) ** 2
        if diff > biggest_diff:
            print(f"biggest diff f({SIGN(x_i):.03f}) = GOLD[{y_gold:.03f}], ~[{y_approx:.03f}], ERR[{abs((y_gold-y_approx)):.03f}]")
            biggest_diff = diff
        xm = x_map(x_i)
        # print(x_i, y_gold, y_approx)
        if current_gold != e_gold:
            if current_gold is not None:
                ax.plot(x_pts, e_pts, marker='.', markersize=dot_size, color='black', label="Baseline")
                ax2.plot(x_pts, m_pts, marker='.', markersize=dot_size, color='black')
            current_gold = e_gold
            for a in [e_pts, x_pts, m_pts]:
                a.clear()
        x_pts.append(xm)
        e_pts.append(e_gold)
        m_pts.append(m_gold)

        if current_approx != e_approx:
            if current_approx is not None:
                if plot_approx:
                    ax.plot(x_approx_pts, e_approx_pts, marker='.', markersize=2, color='red', label="Approx.")
                if not has_done_once:
                    if plot_approx:
                        ax.legend()
                    has_done_once = True
                if plot_approx:
                    ax2.plot(x_approx_pts, m_approx_pts, marker='.', markersize=2, color='red')
            current_approx = e_approx
            for a in [e_approx_pts, x_approx_pts, m_approx_pts]:
                a.clear()
        x_approx_pts.append(xm)
        e_approx_pts.append(e_approx)
        m_approx_pts.append(m_approx / 128)
    ax.plot(x_pts, e_pts, marker='.', markersize=dot_size, color='black', label="Baseline")
    ax2.plot(x_pts, m_pts, marker='.', markersize=dot_size, color='black')
    if plot_approx:
        ax.plot(x_approx_pts, e_approx_pts, marker='.', markersize=2, color='red', label="Approx")
        ax2.plot(x_approx_pts, m_approx_pts, marker='.', markersize=2, color='red')
    if plot_err:
        ax3.plot(err_x_pts, err_pts)
        ax4.plot(err_x_pts, err_real_pts, label="real error")
    mv_x = []
    mv_real = []
    window_sz = 32
    for i in range(window_sz, len(err_real_pts) - window_sz):
        window = err_real_pts[i - window_sz:i + 1:window_sz]
        mv_real.append(sum(window) / len(window))
        mv_x.append(err_x_pts[i])
    # ax4.plot(mv_x, mv_real, label=f"error (moving avg {window_sz})")
    print(f"L1: {errl1 / len(x) * 128}")
    print(f"L2: {(err ** .5) / len(x) * 128}")
    print(f"Average err: {(realerr ** 0.5)/len(x):0.4f}")


x = []
for i in range(x_e_min, x_e_max + 1):
    for j in range(128):
        h = f"0x1.{hex(2 * j)[2:]}p{i}"
        f_ = float.fromhex(h)
        x.append(SIGN(f_))

x.sort()
x = np.array(x)
to_sup_map = {"0": "\u2070", "1": "\u00b9", "2": "\u00b2", "3": "\u00b3",
              "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
              "8": "\u2078", "9": "\u2079", "-": "\u207b"}


def to_sup(a: str):
    if a:
        return to_sup_map[a[0]] + to_sup(a[1:])
    else:
        return ""


if __name__ == "__main__":
    nplots = 2
    if plot_err:
        nplots += 2
    fig, axs = plt.subplots(1, nplots, figsize=(nplots * 2.5, 3), dpi=800)
    ############### EXP + #####################
    ax = axs[0]
    ax2 = axs[1]
    if plot_err:
        ax3 = axs[2]
        ax4 = axs[3]
        ax3.set_title("ERROR (BITS)")
        ax3.set_xlabel("(c)")
        ax4.set_title("ERROR (REAL)")
        ax4.set_xlabel("(d)")
    # ax.set_title("EXPONENT")
    ax.set_xlabel("(a)")
    # ax2.set_title("MANTISSA")
    ax2.set_xlabel("(b)")
    if plot_approx:
        ax.legend()
    plot_discontinuous_f_on(x, get_approx, f)
    fig.tight_layout()
    fig.savefig('nonlinear-figs1.pdf')
