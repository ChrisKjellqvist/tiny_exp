module SRAMBaseline_FP32 (
    input         clk,
    input         rst,
    input  [31:0] x,
    output [31:0] y,

    // width for index is given by [e_adj, mantissa_[hi/lo], sign]
    //                               4b          4b/3b        1b
    // 4b + 11 + 1b
    output [15:0]  sram_hi_idx,
    input  [31:0] sram_hi,
    // 4b + 12 + 1b
    output [16:0]  sram_lo_idx,
    input  [31:0] sram_lo
);

wire       S = x[31];
wire [7:0] E = x[30:23];
wire [22:0] M = x[22:0];

wire [10:0] M_hi = M[22:12];
wire [11:0] M_lo = M[11:0];

localparam Emin = -7;
localparam Emax = 6;

wire [7:0] E_norm = E - (127+Emin);
wire [3:0] E_adj = E_norm[3:0];

assign sram_hi_idx = {E_adj, M_hi, S};
assign sram_lo_idx = {E_adj, M_lo, S};

wire is_big = E >= (127 + Emax);
wire is_small = E < (127 + Emin);

wire [31:0] EXTREME = {1'b0, {8{!S}}, 23'b0};
wire [31:0] ONE = 32'h3f800000;

wire [31:0] approx;
FP_MUL_FP32 mul(
    .clk(clk),
    .rst(rst),
    .inA(sram_hi),
    .inB(sram_lo),
    .out(approx)
);
assign y = is_big ? EXTREME : (is_small ? ONE : approx);
endmodule
