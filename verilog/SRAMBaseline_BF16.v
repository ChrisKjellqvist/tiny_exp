module SRAMBaseline_BF16 (
    input         clk,
    input         rst,
    input  [15:0] x,
    output [15:0] y,

    // width for index is given by [e_adj, mantissa_[hi/lo], sign]
    //                               4b          4b/3b        1b
    output [8:0]  sram_hi_idx,
    input  [15:0] sram_hi,

    output [7:0]  sram_lo_idx,
    input  [15:0] sram_lo
);

wire       S = x[15];
wire [7:0] E = x[14:7];
wire [6:0] M = x[6:0];

wire [3:0] M_hi = M[6:3];
wire [2:0] M_lo = M[2:0];

localparam Emin = -7;
localparam Emax = 6;

wire [7:0] E_norm = E - (127+Emin);
wire [3:0] E_adj = E_norm[3:0];
assign sram_hi_idx = {E_adj, M_hi, S};
assign sram_lo_idx = {E_adj, M_lo, S};

wire is_big = E >= (127 + Emax);
wire is_small = E < (127 + Emin);

wire [15:0] EXTREME = {1'b0, {8{!S}}, 7'b0};
wire [15:0] ONE = 16'h3f80;

wire [15:0] approx;
FP_MUL mul(
    .clk(clk),
    .rst(rst),
    .inA(sram_hi),
    .inB(sram_lo),
    .out(approx)
);
assign y = is_big ? EXTREME : (is_small ? ONE : approx);
endmodule
