module FP_MUL(
  input clk,
  input rst,
  input [15:0] inA,
  input [15:0] inB,
  output [15:0] out
);

fpnew_top #(
    .Features('{
      Width: 16,
      EnableVectors: 0,
      EnableNanBox: 0,
      FpFmtMask: 1,
      IntFmtMask: 0
    }),
    .Implementation('{
      PipeRegs: '{default: 0},
      UnitTypes:  '{'{default: fpnew_pkg::PARALLEL}, // ADDMUL
                    '{default: fpnew_pkg::DISABLED}, // DIVSQRT
                    '{default: fpnew_pkg::DISABLED}, // NONCOMP
                    '{default: fpnew_pkg::DISABLED}},// CONV
      PipeConfig: fpnew_pkg::BEFORE
    })
) top (
  .clk_i(clk),
  .rst_ni(!rst),
  .operands_i('{inA, inB, 0}),
  .rnd_mode_i(fpnew_pkg::RNE),
  .op_i(fpnew_pkg::MUL),
  .op_mod_i(0),
  .src_fmt_i(fpnew_pkg::FP16ALT),
  .dst_fmt_i(fpnew_pkg::FP16ALT),
  .int_fmt_i(fpnew_pkg::INT16),
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