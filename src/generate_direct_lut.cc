//
// Created by Christopher Kjellqvist on 1/16/24.
//

#include <cstdio>
#include <cmath>
#include <bit>

auto prefix = "module Exp_LUT(\n"
              "    input clock,\n"
              "    input reset,\n"
              "    input enable,\n"
              "    input [15:0] x,\n"
              "    output [15:0] y\n"
              ");\n"
              "\n"
              "(* rom_style = \"distributed\" *) reg [15:0] y_r;\n"
              "assign y = y_r;\n"
              "wire [7:0] exp = x[14:7];\n"
              "wire sign = x[15];"
              "always @(posedge clock) begin\n"
              "   if (enable)\n"
              "   begin\n"
              "       if (exp >= 134)\n"
              "       begin\n"
              "           if (sign) begin\n"
              "               y_r <= 0;\n"
              "           end else begin\n"
              "               y_r <= 16'h7F80;\n"
              "           end\n"
              "       end else begin\n"
              "            case (x)\n";
auto tab = "              ";

int main() {
  FILE *f = fopen("Exp_LUT.v", "w");
  fprintf(f, "%s\n", prefix);
  for (int exp = 0; exp <= 133; ++exp) {
    for (int mant = 0; mant < 128; ++mant) {
      for (int sign = 0; sign < 2; ++sign) {
        uint32_t raw = (sign << 31) | (exp << 23) | (mant << 16);
        float x_raw = reinterpret_cast<float &>(raw);
        float y_raw = std::exp(x_raw);
        uint32_t y_raw_u32 = std::bit_cast<uint32_t>(y_raw);
        y_raw_u32 &= 0xFFFF0000L;
        uint32_t y_raw_bits = reinterpret_cast<uint32_t &>(y_bf16);
        fprintf(f, "%s16'h%04x: y_r <= 16'h%04x;\n", tab, raw >> 16, y_raw_u32 >> 16);
      }
    }
  }
  fprintf(f, "            endcase\n"
             "        end"
             "    end\n"
             "end\n"
             "endmodule\n");
  fclose(f);
  return 0;
}
