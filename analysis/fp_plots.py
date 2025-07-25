import matplotlib.pyplot as plt
import numpy as np
import torch as to
import math as ma
import enum


class Flavor(enum.Enum):
    CPP = 0
    VERILOG = 1

htable = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
          '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
          'e': 14, 'f': 15}

def exp(x):
    return np.exp(x)


def gen_htan(x, L, k, c, z):
    return L / (1 + np.exp(-k * (x - c))) - z


def silu_sig(x):
    return gen_htan(x, L=1, k=1, c=0, z=0)  

def htan(x):
    return gen_htan(x, L=2, k=2, c=0, z=1)

def gelu_sig(x):
    return gen_htan(x, 1, 1.6, 0, 0)

def sig(x):
    return gen_htan(x, 1, 1, 0, 0)

def recip(x):
    return 1.0 / ma.sqrt(x)

def sqrt(x):
    return ma.sqrt(x)

def div(x):
    return 1/x

def identity(x):
    return x

def silu(x):
    return silu_sig(x)

def gelu(x):
    return gelu_sig(x)

does_not_support_zero = [recip, sqrt, div]
does_not_support_neg = [recip, sqrt]
does_not_support_pos = [silu]

mantissa_used_as_index = 2
mantissa_wid = 23
mantissa_tot = 1 << mantissa_wid
e_wid:int = 8
f = silu
x_e_min = -7
x_e_max = 5
plot_chart = True
plot_err = True
plot_approx = True
adjust_curve = True
flavor=Flavor.VERILOG
err = 0
errl1 = 0
err_real = 0
errl2_real = 0
N = 0
biggest_diff = 0


# SIGN = lambda __x:  -__x
base_signs = []
if f not in does_not_support_pos:
    base_signs.append(lambda __x: __x)
if f not in does_not_support_neg:
    base_signs.append(lambda __x: -__x)
for SIGN in base_signs:
    is_neg = SIGN(1) == -1
    cpp_bases = ""
    cpp_offsets = ""

    if flavor==Flavor.CPP:
        cpp_bases = "const int32_t bases[] = {"
        cpp_offsets = "const int64_t offsets[] = {"

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


    def to_flt_hex_str(e, m):
        flt_adjusted_e: int = int(e) + ((1 << (e_wid-1))-1)
        flt_adjusted_e_shift: int = flt_adjusted_e << mantissa_wid
        flt_adjusted_m: int = int(m*mantissa_tot)
        flt_combined_raw: int = flt_adjusted_e_shift | flt_adjusted_m
        return hex(flt_combined_raw)[2:]


    _, lim_y_e, lim_y_m = unpack_float(f(SIGN(2 ** x_e_min)))
    d_lookup = {}
    for x_ in range(x_e_min, x_e_max + 1):
        for m_ in range(0, 1<<mantissa_used_as_index):
            current = SIGN((2 ** x_)*(1.0 + m_ / (1 << mantissa_used_as_index)))
            if m_ == (1 << mantissa_used_as_index) - 1:
                next = SIGN(2 ** (x_ + 1))
            else:
                next = SIGN((2 ** x_)*(1.0 + (m_+1) / (1 << mantissa_used_as_index)))
            _, flo_e, flo_m = unpack_float(f(current))
            _, fhi_e, fhi_m = unpack_float(f(next))
            print(f"// {x_}: {flo_m:04f} {fhi_m:04f}")
            # print(current, " ", next)
            diff_frac = fhi_e - flo_e + fhi_m - flo_m
            diff_large = diff_frac * (mantissa_tot << mantissa_used_as_index)
            diff_int = int(diff_large)
            d_lookup[(x_, m_)] = (diff_large, (flo_e, flo_m * mantissa_tot))

            BASE_FLT = to_flt_hex_str(flo_e, flo_m)

            if abs(diff_int) > 0:
                OFFSET_SZ = ma.log2(abs(diff_int))
                OFFSET_SZ_RND = int(ma.ceil(OFFSET_SZ))
            else:
                OFFSET_SZ_RND = 0
                print("DIFF @ x_ = " + str(diff_int))
            L = 1 + 8 + mantissa_wid
            if flavor == Flavor.VERILOG:
                print(f"// {OFFSET_SZ_RND}-bits")
                idx = 0 if SIGN(-1)<0 else 1
                idx2 = x_-x_e_min
                print(f"assign BASES  [{0 if SIGN(-1)<0 else 1}][{x_-x_e_min:2d}] = 16'h{BASE_FLT};")
                if SIGN(-1)>0:
                    ans = hex(diff_int)[3:]
                else:
                    ans = hex(diff_int)[2:]
                if ans == "":
                    ans = "0"
                if SIGN(-1)>0:
                    ans = f"~(26'h{ans})+1"
                else:
                    ans = f"26'h{ans}"
                print(f"assign OFFSETS[{idx}][{idx2:2d}] = {ans};")
            elif flavor == Flavor.CPP:
                assert mantissa_used_as_index == 0
                idx = 0 if SIGN(-1)<0 else 1
                idx2 = x_-x_e_min
                comma = ", " if x_ != x_e_max else ""
                cpp_bases += f"0x{BASE_FLT}{comma}"
                if SIGN(-1)>0:
                    ans = hex(diff_int)[3:]
                else:
                    ans = hex(diff_int)[2:]
                if ans == "":
                    ans = "0"
                ans = f"0x{ans}"
                cpp_offsets += f"{ans}{comma}"
    if flavor == Flavor.CPP:
        cpp_offsets += "};"
        cpp_bases += "};"
        if f not in does_not_support_zero:
            at_zero = f(0)
            _, e, m = unpack_float(at_zero)
            print(e, ' ', m)
            print(f"const int32_t default_at_0 = 0x{to_flt_hex_str(e, m)};")
            print(cpp_bases)
            print(cpp_offsets)


    # for x_ in range(x_e_min, x_e_max  + 1):
    #     _, e, m = unpack_float(f(SIGN(2 ** x_)))
    #     q = hex((int(e) + mantissa_tot - 1) * mantissa_tot + int(m * mantissa_tot))[2:]
    #     print(x_, ": ", e, m * mantissa_tot, f"16'h{q}")

    # for key in d_lookup.keys():
    #     print(key, ": ", d_lookup[key] * mantissa_tot)

    def get_approx(x):
        _, e, m = unpack_float(x)
        top_m = int(m * mantissa_tot) >> (mantissa_wid-mantissa_used_as_index)
        delta, (base_e, base_m) = d_lookup[(e, top_m)]
        sub_correct_for_extra_mantissa = (1 / (1 << mantissa_used_as_index)) * top_m
        # print(top_m, " ", sub_correct_for_extra_mantissa, " ", m)
        delta_m = m - sub_correct_for_extra_mantissa 
        whole = delta * delta_m + base_m
        int_whole = int(whole)
        rem = whole - int_whole
        m_app = int(int_whole % mantissa_tot) + (1 if rem >= 0.5 else 0)
        e_app = base_e + int_whole // mantissa_tot
        if m_app < mantissa_tot:
            return False, e_app, m_app
        else:
            return False, e_app + 1, 0


    def plot_discontinuous_f_on(x, fapprox, fgold):
        xticks = []
        xtick_num = []
        global err
        global errl1
        global err_real
        global errl2_real
        global N
        global biggest_diff

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
            # ax3.grid(True)
            ax4.grid(True)
            # ax3.set_xticks(xtick_num, labels=xticks)
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

            if f == silu or f == gelu:
                real_approx_neg = y_approx * x_i_rnd
                real_approx_pos = x_i_rnd - x_i_rnd * y_approx
                real_gold_neg = x_i_rnd * y_gold
                real_gold_pos = x_i_rnd - x_i_rnd * y_gold
                err_relA = np.abs((real_approx_neg - real_gold_neg)/real_gold_neg)
                err_relB = np.abs((real_approx_pos - real_gold_pos)/real_gold_pos)

                errl1 += err_relA + err_relB
                err_real += np.abs(real_approx_neg - real_gold_neg) + np.abs(real_gold_pos-real_approx_pos)
                errl2_real += np.abs(real_approx_neg - real_gold_neg)**2 + np.abs(real_gold_pos-real_approx_pos)**2
                err += err_relA ** 2 + err_relB ** 2

                err_rel = err_relA
                # print(err_rel)
                N += 2
            else:
                err_rel = 100 * (y_approx - y_gold)/y_gold
                errl1 += np.abs(err_rel)
                err_real += np.abs(y_approx - y_gold)
                errl2_real += (y_approx - y_gold) ** 2
                err += err_rel ** 2
                N += 1

            err_pts.append(diff_bits * mantissa_tot)
            err_real_pts.append(err_rel)

            if abs(err_rel) > biggest_diff:
                # print(f"biggest diff f({SIGN(x_i):.03f}) = GOLD[{y_gold:.03f}], ~[{y_approx:.03f}], ERR[{err_rel:.03f}]")
                biggest_diff = abs(err_rel)
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
            # ax3.plot(err_x_pts, err_pts)
            ax4.plot(err_x_pts, err_real_pts, label="real error")
        if plot_approx and plot_chart:
            ax.legend([base_line[0], approx_line[0]], ["Baseline", "Approx."])


    x = []
    skip_by = 1 if mantissa_wid < 12 else 1<< (mantissa_wid - 14)
    for i in range(x_e_min, x_e_max + 1):
        for j in range(0, mantissa_tot, skip_by):
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


    nplots = 0
    if plot_chart:
        space_per_plot = 2.5
        nplots += 2
    else:
        space_per_plot = 3.5
    if plot_err:
        nplots += 1
    # fig, axs = plt.subplots(nplots, 1, figsize=(3.3, nplots * space_per_plot / 1.4), dpi=800)
    fig, axs = plt.subplots(1, nplots, figsize=(nplots * space_per_plot, 2.5), dpi=800)

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
        # ax3 = axs[i]
        if nplots == 1:
            ax4 = axs
        else:
            ax4 = axs[i]
        # ax3.set_ylabel("Error (bit-distance)")
        ax4.set_ylabel("Error (%)")

        # ax3.set_xlabel(sublabels[i])
        if nplots > 1:
            ax4.set_xlabel(sublabels[i+1])
    plot_discontinuous_f_on(x, get_approx, f)
    fig.tight_layout()
    ending = "png"
    inner = 'neg.' if is_neg else 'pos.'
    print(f"WROTE TO nonlinear-figs1.{inner}{f.__name__}.{ending}")
    fig.savefig(f'nonlinear-figs1.{inner}{f.__name__}.{ending}')

print(f"L1: {errl1 / N}")
print(f"L2: {(err ** .5) / N}")
print(f"L1-real: {err_real / N: .04f}")
print(f"L2-norm: {ma.pow(errl2_real, 0.5):.04f}")
print(f"RMSE: {ma.pow(errl2_real, 0.5) / N: .04f}")
print(f"Tested on {N} points")
print(f"Highest relative error: {biggest_diff}%")

