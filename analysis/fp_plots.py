import matplotlib.pyplot as plt
import numpy as np
import torch as to
import math as ma

# mantissa_used_as_index = 0
mantissa_wid = 14
mantissa_tot = 1 << mantissa_wid

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

def sig(x):
    return gen_htan(x, 1, 1, 0, 0)

# proposed uarch:
#   FLOAT( LUT_base[E] + M * LUT_off[E] ) + MUX(?, REG, 0)


f = exp
SIGN = lambda __x:  __x
x_e_min = -7
x_e_max = 6

plot_chart = True
plot_err = True
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

shamt = 0 if mantissa_wid % 4 == 0 else 4 - (mantissa_wid % 4)
if mantissa_wid % 4 == 0:
    hex_digs = mantissa_wid 
else:
    hex_digs = (mantissa_wid + shamt)//4

def zeropad_shift_and_hexify_mantissa(q):
    assert q >= 0
    q_shift = q << shamt
    hx = hex(q_shift)[2:]
    if len(hx) < hex_digs:
        hx = '0'* (hex_digs - len(hx)) + hx
    return hx
    
def unpack_float(q, verbose=False):
    sign = q < 0
    h = float(q).hex()
    dot = h.find('.')
    pspot = h.find('p')
    mantissa = h[dot + 1:min(dot + hex_digs+1, pspot)]
    if verbose:
        print(hex_digs, " ", mantissa)
    inty = hex2int(mantissa)
    mantissa = (inty // (1 << shamt)) / mantissa_tot
    exponent = int(h[h.find('p') + 1:])
    if verbose:
        print(f"{q} -> {h}, {inty}, {(sign,exponent,mantissa)}")
    return sign, exponent, mantissa

def repack_float(s, e, m, verbose=False):
    s_str = "-" if s else ""
    # align 4 to align with python's float hex string format
    m_big = m * mantissa_tot
    m_int = int(m_big) + (1 if m_big - int(m_big) > 0.5 else 0)
    q = zeropad_shift_and_hexify_mantissa(m_int)
    m_str = "1." + q
    flt = f"{s_str}{m_str}p{int(e)}"
    if verbose:
        print((s, e, m), " -> ", flt, " ", float.fromhex(flt))
    return float.fromhex(flt)


def x_map(x):
    _, e, m = unpack_float(x)
    return e+m


def x_map_ar(x):
    return np.array([x_map(xp) for xp in x])


_, lim_y_e, lim_y_m = unpack_float(f(SIGN(2 ** x_e_min)))
d_lookup = {}
for x_ in range(x_e_min, x_e_max + 1):
    _, flo_e, flo_m = unpack_float(f(SIGN(2 ** x_)))
    _, fhi_e, fhi_m = unpack_float(f(SIGN(2 ** (x_ + 1))))
    diff_frac = fhi_e - flo_e + fhi_m - flo_m
    diff_large = diff_frac * mantissa_tot
    diff_int = int(diff_large)
    d_lookup[x_] = diff_frac
    BASE_FLT = hex((int(flo_e) + (mantissa_tot-1)) * mantissa_tot + int(flo_m * mantissa_tot))[2:]
    if diff_int > 0:
        OFFSET_SZ = ma.log2(abs(diff_int))
        OFFSET_SZ_RND = int(ma.ceil(OFFSET_SZ))
        print(f"// {OFFSET_SZ_RND}-bits")
    L = 1 + 8 + mantissa_wid
    print(f"assign BASES  [{0 if SIGN(-1)<0 else 1}][{x_-x_e_min:2d}] = 16'h{BASE_FLT};")
    if SIGN(-1)>0:
        print(f"assign OFFSETS[{0 if SIGN(-1)<0 else 1}][{x_-x_e_min:2d}] = ~(26'h{hex(diff_int)[3:]})+1;")
    else:
        print(f"assign OFFSETS[{0 if SIGN(-1)<0 else 1}][{x_-x_e_min:2d}] = 26'h{hex(diff_int)[2:]};")

# for x_ in range(x_e_min, x_e_max  + 1):
#     _, e, m = unpack_float(f(SIGN(2 ** x_)))
#     q = hex((int(e) + mantissa_tot - 1) * mantissa_tot + int(m * mantissa_tot))[2:]
#     print(x_, ": ", e, m * mantissa_tot, f"16'h{q}")

# for key in d_lookup.keys():
#     print(key, ": ", d_lookup[key] * mantissa_tot)

def get_approx(x):
    _, e, m = unpack_float(x)
    su = lim_y_m
    for i in range(x_e_min, e):
        su += d_lookup[i]
    su += d_lookup[e] * m
    assert (m < 1)
    whole = su * mantissa_tot
    int_whole = int(whole)
    rem = whole - int_whole
    m_app = int(int_whole % mantissa_tot) + (1 if rem >= 0.5 else 0)
    if m_app < mantissa_tot:
        e_app = np.floor(lim_y_e + su)
        return False, e_app, m_app
    else:
        return False, np.floor(lim_y_e + su) + 1, 0
    # print("APPROX: ", e_app, m_app)


def plot_discontinuous_f_on(x, fapprox, fgold):
    xticks = []
    xtick_num = []
    every_other = True
    for i in range(x_e_min, x_e_max + 2):
        xtick_num.append(i)
        if every_other:
            xticks.append(f"2{to_sup(str(i))}")
        else:
            xticks.append("")
        every_other = not every_other
    if plot_chart:
        ax.set_xticks(xtick_num, labels=xticks)
        ax2.set_xticks(xtick_num, labels=xticks)
        # ax.set_yticks(range(-50, 50))
        # ax2.set_yticks(np.arange(0, 1, 1.0 / 128), labels="")
        ax.grid(True)
        ax2.grid(True)
    if plot_err:
        ax3.grid(True)
        ax4.grid(True)
        ax3.set_xticks(xtick_num, labels=xticks)
        ax4.set_xticks(xtick_num, labels=xticks)

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
    N = 0
    biggest_diff = 0
    has_done_once = False
    approx_line = None
    base_line = None

    for x_i in x:
        x_i_s, x_i_e, x_i_m = unpack_float(x_i, verbose=False)
        x_i_rnd = repack_float(x_i_s, x_i_e, x_i_m, verbose=False)
        # assert abs(x_i - x_i_rnd) <= (1 << x_i_e) / (1 << mantissa_wid)
        y_gold = fgold(x_i_rnd)
        _, e_approx, m_approx = fapprox(x_i_rnd)
        s, e_gold, m_gold = unpack_float(y_gold)
        y_approx = repack_float(s, e_approx, m_approx / mantissa_tot)
        diff_bits = np.abs((e_approx + m_approx / mantissa_tot) - (e_gold + m_gold))

        err_rel = 100 * np.abs(y_approx - y_gold)/y_gold
        errl1 += err_rel
        err += err_rel ** 2
        N += 1

        err_pts.append(diff_bits * mantissa_tot)

        err_real_pts.append(err_rel)

        if err_rel > biggest_diff:
            # print(f"biggest diff f({SIGN(x_i):.03f}) = GOLD[{y_gold:.03f}], ~[{y_approx:.03f}], ERR[{err_rel:.03f}]")
            biggest_diff = err_rel
        xm = x_map(x_i)
        err_x_pts.append(xm)
        # print(x_i, y_gold, y_approx)
        if current_gold != e_gold:
            if current_gold is not None and plot_chart:
                l = ax.plot(x_pts, e_pts, marker='.', markersize=dot_size, color='black')
                if base_line is None:
                    base_line = l
                ax2.plot(x_pts, m_pts, marker='.', markersize=dot_size, color='black')
            current_gold = e_gold
            for a in [e_pts, x_pts, m_pts]:
                a.clear()
        x_pts.append(xm)
        e_pts.append(e_gold)
        m_pts.append(m_gold)

        if current_approx != e_approx:
            if current_approx is not None:
                if plot_approx and plot_chart:
                    if not has_done_once:
                        l = ax.plot(x_approx_pts, e_approx_pts, marker='.', markersize=2, color='red')
                        if approx_line is None:
                            approx_line = l
                        ax2.plot(x_approx_pts, m_approx_pts, marker='.', markersize=2, color='red')
                        has_done_once = True
                    else:
                        ax.plot(x_approx_pts, e_approx_pts, marker='.', markersize=2, color='red')
                        ax2.plot(x_approx_pts, m_approx_pts, marker='.', markersize=2, color='red')
                        
            current_approx = e_approx
            for a in [e_approx_pts, x_approx_pts, m_approx_pts]:
                a.clear()
        x_approx_pts.append(xm)
        e_approx_pts.append(e_approx)
        m_approx_pts.append(m_approx / mantissa_tot)
    if plot_chart:
        l = ax.plot(x_pts, e_pts, marker='.', markersize=dot_size, color='black')
        if base_line is None:
            base_line = l
        ax2.plot(x_pts, m_pts, marker='.', markersize=dot_size, color='black')
        if plot_approx:
            l = ax.plot(x_approx_pts, e_approx_pts, marker='.', markersize=2, color='red')
            if approx_line is None:
                approx_line = l

            ax2.plot(x_approx_pts, m_approx_pts, marker='.', markersize=2, color='red')
    if plot_err:
        ax3.plot(err_x_pts, err_pts)
        ax4.plot(err_x_pts, err_real_pts, label="real error")
    if plot_approx and plot_chart:
        ax.legend([base_line[0], approx_line[0]], ["Baseline", "Approx."])
    print(f"L1: {errl1 / N}")
    print(f"L2: {(err ** .5) / N}")
    print(f"Tested on {N} points")
    # print(f"biggest diff f({SIGN(x_i):.03f}) = GOLD[{y_gold:.03f}], ~[{y_approx:.03f}], ERR[{err_rel:.03f}]")


x = []
for i in range(x_e_min, x_e_max + 1):
    for j in range(mantissa_tot):
        h = f"0x1.{zeropad_shift_and_hexify_mantissa(j)}p{i}"        
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
    nplots = 0
    if plot_chart:
        space_per_plot = 2.5
        nplots += 2
    else:
        space_per_plot = 3.5
    if plot_err:
        nplots += 2
    # fig, axs = plt.subplots(nplots, 1, figsize=(3.3, nplots * space_per_plot), dpi=800)
    fig, axs = plt.subplots(1, nplots, figsize=(nplots * space_per_plot, 3), dpi=800)

    ############### EXP + #####################
    i = 0
    sublabels = ["(a)", "(b)", "(c)", "(d)"]
    if plot_chart:
        ax = axs[i]
        ax2 = axs[i+1]
        ax2.set_ylim([0, 1])
        ax.set_xlabel(sublabels[i])
        ax2.set_xlabel(sublabels[i+1])
        i += 2
    if plot_err:
        ax3 = axs[i]
        ax4 = axs[i+1]
        ax3.set_ylabel("Error (bit-distance)")
        ax4.set_ylabel("Error (%)")

        ax3.set_xlabel(sublabels[i])
        ax4.set_xlabel(sublabels[i+1])
    plot_discontinuous_f_on(x, get_approx, f)
    fig.tight_layout()
    ending = "pdf" if mantissa_wid < 10 else "png"
    fig.savefig(f'nonlinear-figs1.{ending}')
    
