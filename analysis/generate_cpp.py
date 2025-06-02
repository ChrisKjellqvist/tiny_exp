import torch
import numpy as np
import math

em_pairs = [(8, 7), (8, 23)]

htable = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
          '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
          'e': 14, 'f': 15}


def hex2int(h, acc=0):
    if len(h) == 0:
        return acc
    else:
        return hex2int(h[1:], 16 * acc + htable[h[0]])

if __name__ == "__main__":
    hptr = open("cpp/tiny_exp.h", 'w')
    sptr = open("cpp/tiny_exp.cc", 'w')
    hptr.write("""#ifndef TINY_EXP_H
#define TINY_EXP_H
#include <cinttypes>
""")
    sptr.write('#include "tiny_exp.h"\n#include <bit>\n')
    for E, M in em_pairs:
        Mdivs = 2 ** M
        Eoff = (2 ** (E - 1)) - 1

        def unpack_float(q):
            sign = q < 0
            h = float(q).hex()
            dot = h.find('.')
            pspot = h.find('p')
            mantissa = h[dot + 1:min(dot + 1 + int(np.ceil(float(M)/4)), pspot)]
            # if we have a odd-width'd mantissa, need to shift it some
            intdiv = 8 - (M % 8)
            mrenorm = (hex2int(mantissa) >> intdiv) / Mdivs
            exponent = int(h[h.find('p') + 1:])
            return 1 if sign else 0, exponent, mrenorm


        def repack_float(s, e, m, verbose=False):
            s_str = "-" if s else ""
            intdiv = 8 - (M % 8)
            q = hex(int(m * (1 << intdiv) * Mdivs))[2:]
            if len(q) == 1:
                q = "0" + q
            m_str = "1." + q
            flt = f"{s_str}{m_str}p{int(e)}"
            if verbose:
                print(flt)
            return float.fromhex(flt)


        def f(x):
            return np.exp(x)


        def get_approx(x):
            if x == float("-inf"):
                return 0
            elif x == float("inf"):
                return float("inf")
            elif np.isnan(x):
                raise Exception(f"Received exp(nan): {x}")
            s, e, m = unpack_float(x)
            assert (m < 1)
            if e < x_e_min:
                return 1
            if e > x_e_max:
                if s:
                    return 0
                else:
                    return torch.inf
            e_acc, m_acc = b_lookup[(s, e)]
            c = d_lookup[(s, e)]
            m_acc += c * m
            e_app = np.floor(e_acc + m_acc)
            # print(x, c, s, e, m, e_acc, m_acc)
            m_app = int((m_acc * Mdivs) % Mdivs)
            q = repack_float(False, e_app, m_app)
            if np.isnan(q):
                raise Exception(f"Returning exp({x})=nan")
            return q


        x_e_max = int(np.log2(2 ** (E - 1) * np.log(2)))
        # this is probably good enough
        x_e_min = -x_e_max-1
        cdat_ty = f"uint{int(2**np.ceil(np.log2(1+E+M)))}_t"
        cdat_ty_large = f"uint{int(2**np.ceil(np.log2(1+E+2*M)))}_t"
        ftype = "__fp16" if cdat_ty == "uint16_t" else "float"
        blookup_str = f"""const {cdat_ty} blookup[2][{x_e_max - x_e_min + 1}] = {{"""
        dlookup_str = f"""const {cdat_ty_large} dlookup[2][{x_e_max - x_e_min + 1}] = {{"""
        d_lookup = {}
        b_lookup = {}
        for sign in [0, 1]:
            blookup_str += "\n\t{"
            dlookup_str += "\n\t{"
            def SIGN(x):
                if sign == 0:
                    return x
                else:
                    return -x
            # _, lim_y_e, lim_y_m = unpack_float(f(SIGN(2 ** x_e_min)))

            for x_ in range(x_e_min, x_e_max):
                _, flo_e, flo_m = unpack_float(f(SIGN(2 ** x_)))
                _, fhi_e, fhi_m = unpack_float(f(SIGN(2 ** (x_ + 1))))
                # BASE
                q = hex((int(flo_e) + Eoff) * Mdivs + int(flo_m * Mdivs))
                b_lookup[(sign, x_)] = (flo_e, flo_m)
                blookup_str += q
                # DELTA
                d = fhi_e - flo_e + fhi_m - flo_m
                d_lookup[(sign, x_)] = d
                print(sign, x_, d)
                dlookup_str += hex(int(np.abs(d) * Mdivs))
                if x_ != x_e_max-1:
                    blookup_str += ", "
                    dlookup_str += ", "
            blookup_str += "}"
            dlookup_str += "}"
            if sign == 0:
                blookup_str += ","
                dlookup_str += ","
            else:
                blookup_str += "};"
                dlookup_str += "};"
        EMASK = hex((2**E)-1)
        MMASK = hex((2**M)-1)
        ONE = hex(((2**(E-1))-1)*(2**M))
        sptr.write(f"""
namespace fpE{E}M{M} {{
{cdat_ty} tiny_exp(const {cdat_ty} &as_bit) {{
    {cdat_ty} S = as_bit >> {E+M};
    {cdat_ty} E = (as_bit >> {M}) & {EMASK};
    if (E >= {x_e_max+Eoff}) {{
        if (S) return 0;
        else return {hex(((2**E)-1)*2**M)};
    }} else if (E < {x_e_min+Eoff}) {{
        return {ONE};
    }}
    {cdat_ty} Enorm = (E - {x_e_min + Eoff});
    {cdat_ty_large} M = (as_bit & {MMASK}) * dlookup[S][Enorm];
    if (S) {{
        return blookup[S][Enorm] - static_cast<{cdat_ty}>(M >> {M});
    }} else {{
        return blookup[S][Enorm] + static_cast<{cdat_ty}>(M >> {M});
    }}
}}
float tiny_exp(const float &f) {{
    uint32_t q = std::bit_cast<uint32_t>(f);
    {cdat_ty} r = tiny_exp(static_cast<{cdat_ty}>(q >> {32-(1+E+M)}));
    return std::bit_cast<float>(static_cast<uint32_t>(r)<<{32-(1+E+M)});
}}
}}""")
        hptr.write(f"""
namespace fpE{E}M{M} {{
float tiny_exp(const float &f);
{cdat_ty} tiny_exp(const {cdat_ty} &as_bit);
{blookup_str}
{dlookup_str}
}}""")
    hptr.write("\n#endif")
    sptr.close()
    hptr.close()
