module FP_INTCAST_BF16(
  input clk,
  input rst,
  input [15:0] inA,
  output [15:0] out
);

fpnew_top #(
    .Features('{
      Width: 32,
      EnableVectors: 0,
      EnableNanBox: 0,
      FpFmtMask: 5'b10000,
      IntFmtMask: 5'b10000
    }),
    .Implementation('{
      PipeRegs: '{default: 0},
      UnitTypes:  '{'{default: fpnew_pkg::DISABLED}, // ADDMUL
                    '{default: fpnew_pkg::DISABLED}, // DIVSQRT
                    '{default: fpnew_pkg::DISABLED}, // NONCOMP
                    '{default: fpnew_pkg::MERGED}},// CONV
      PipeConfig: fpnew_pkg::BEFORE
    })
) top (
  .clk_i(clk),
  .rst_ni(!rst),
  .operands_i({inA, inA, inA}),
  .rnd_mode_i(fpnew_pkg::RNE),
  .op_i(fpnew_pkg::F2I),
  .op_mod_i(0),
  .src_fmt_i(fpnew_pkg::FP32),
  .dst_fmt_i(fpnew_pkg::FP32),
  .int_fmt_i(fpnew_pkg::INT32),
  .vectorial_op_i(0),
  .tag_i(0),
  .simd_mask_i(1),
  .in_valid_i(1),
  .in_ready_o(),
  .flush_i(0),
  .result_o(out),
  .status_o(),
  .tag_o(),
  .out_valid_o(),
  .out_ready_i(1),
  .busy_o()
);

endmodule
