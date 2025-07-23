module FP_FMA_FP32(
  input clk,
  input rst,
  input [31:0] in_mul0,
  input [31:0] in_mul1,
  input [31:0] in_add,
  output [31:0] out
);

fpnew_top #(
    .Features('{
      Width: 32,
      EnableVectors: 0,
      EnableNanBox: 0,
      FpFmtMask: 5'b10000,
      IntFmtMask: 0
    }),
    .Implementation('{
      PipeRegs: '{default: 0},
      UnitTypes:  '{'{default: fpnew_pkg::MERGED}, // ADDMUL
                    '{default: fpnew_pkg::DISABLED}, // DIVSQRT
                    '{default: fpnew_pkg::DISABLED}, // NONCOMP
                    '{default: fpnew_pkg::DISABLED}},// CONV
      PipeConfig: fpnew_pkg::BEFORE
    })
) top (
  .clk_i(clk),
  .rst_ni(!rst),
  .operands_i({in_mul0, in_mul1, in_add}),
  .rnd_mode_i(fpnew_pkg::RNE),
  .op_i(fpnew_pkg::FMADD),
  .op_mod_i(1'b0),
  .src_fmt_i(fpnew_pkg::FP32),
  .dst_fmt_i(fpnew_pkg::FP32),
  .int_fmt_i(fpnew_pkg::INT32),
  .vectorial_op_i(1'b0),
  .tag_i(),
  .simd_mask_i(1'b0),
  .in_valid_i(1'b1),
  .in_ready_o(),
  .flush_i(1'b0),
  .result_o(out),
  .status_o(),
  .tag_o(),
  .out_valid_o(),
  .out_ready_i(1'b1),
  .busy_o()
);

endmodule
