import torch
import numpy as np
import math
import enum

em_pairs = [(8, 7), (8, 23), (5, 10)]

htable = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
          '7': 7, '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13,
          'e': 14, 'f': 15}


class Backend(enum.Enum):
    CUDA = 0
    METAL = 1
    CPU = 2


def hex2int(h, acc=0):
    if len(h) == 0:
        return acc
    else:
        return hex2int(h[1:], 16 * acc + htable[h[0]])

def export_for_backend(backend, f_src):
    sptr = open(f_src, 'w')
    f_src.write("""
#include <cinttypes>

template <typename t>
struct base_d_pr {
    t base, d;
};
""")
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
        if backend == Backend.CPU:
            cdat_ty = f"int{int(2**np.ceil(np.log2(1+E+M)))}_t"
            cdat_ty_large = f"int{int(2**np.ceil(np.log2(1+E+2*M)))}_t"
            array_annot = 'const'
        elif backend == Backend.CUDA:
            raise Exception("CUDA NOT IMPLEMENTED YET")
        elif backend == Backend.METAL:
            array_annot = 'constant'
            l = E+M+1
            if l == 8:
                cdat_ty = 'char4'
                cdat_ty_large = 'short4'
            elif l == 16:
                cdat_ty = 'short4'
                cdat_ty_large = 'int4'
            elif l == 32:
                cdat_ty = 'int4'
                cdat_ty_large = 'long4'

        lookup_str = f"""{array_annot} base_d_pr<{cdat_ty}> lookup[2][{x_e_max - x_e_min + 1}] = {{"""
        d_lookup = {}
        b_lookup = {}
        for sign in [0, 1]:
            lookup_str += "\n\t{"
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
                # DELTA
                d = fhi_e - flo_e + fhi_m - flo_m
                d_lookup[(sign, x_)] = d
                print(sign, x_, d)
                delta = int(np.abs(d) * Mdivs)
                lookup_str += "{" + q + "," + hex(delta) + "}"
                if x_ != x_e_max-1:
                    lookup_str += ", "
            lookup_str += "}"
            if sign == 0:
                lookup_str += ","
            else:
                lookup_str += "};"
        EMASK = hex((2**E)-1)
        MMASK = hex((2**M)-1)
        ONE = hex(((2**(E-1))-1)*(2**M))
        DSHAMT = int(2**np.ceil(np.log2(1+E+M)))
        BMASK = int(2 ** DSHAMT - 1)
        ftype = "__fp16" if cdat_ty == "uint16_t" else "float"
        sptr.write(f"""
namespace fpE{E}M{M} {{
template<>
{cdat_ty} tiny_exp<{cdat_ty}>(const {cdat_ty} &as_bit) {{
    {cdat_ty} S = as_bit >> {E+M};
    {cdat_ty} E = (as_bit >> {M}) & {EMASK};
    if (E >= {x_e_max+Eoff}) [[unlikely]] {{
        if (S) return 0;
        else return {hex(((2**E)-1)*2**M)};
    }}
    if (E < {x_e_min+Eoff}) [[unlikely]] {{
        return {ONE};
    }}
    {cdat_ty} Enorm = E - {x_e_min + Eoff};
    {cdat_ty_fused} fused = lookup[S][Enorm];
    {cdat_ty_large} D = fused >> {DSHAMT};
    {cdat_ty} base = fused & {hex(BMASK)};
    {cdat_ty_large} M = (as_bit & {MMASK}) * D;
    return base + (1 - (S << 1)) * static_cast<{cdat_ty}>(M >> {M});
}}
template<>
float tiny_exp<float>(const float &f) {{
    uint32_t q = std::bit_cast<uint32_t>(f);
    {cdat_ty} r = tiny_exp(static_cast<{cdat_ty}>(q >> {32-(1+E+M)}));
    return std::bit_cast<float>(static_cast<uint32_t>(r)<<{32-(1+E+M)});
}}
}}""")
        hptr.write(f"""
namespace fpE{E}M{M} {{
template <typename t>
t tiny_exp(const t &f);
{lookup_str}
}}""")
    hptr.write("\n#endif")
    sptr.close()
    hptr.close()
