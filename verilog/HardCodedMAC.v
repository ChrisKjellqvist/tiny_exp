module RegFileMAC_BF16 (
    input         clk,
    input         rst,
    input  [15:0] x,
    output [15:0] y
);

wire [15:0] BASES   [0:1][0:12];
wire [15:0] OFFSETS [0:1][0:12];
`include "ExpLUT.v"

wire       S = x[15];
wire [7:0] E = x[14:7];
wire [6:0] M = x[6:0];

localparam Emin = -7;
localparam Emax = 6;

wire [7:0] E_adj = E + (127+Emin);

wire [15:0] base = BASES[S][E_adj];
wire [15:0] offset = OFFSETS[S][E_adj];

wire [22:0] product = M * offset;
wire [15:0] product_used = product[22:7];

wire [15:0] approx = base + product_used;

wire is_big = E >= (127 + Emax);
wire is_small = E < (127 + Emin);

// large negative value -> 0
// large positive value -> 0x7f80;
wire [15:0] EXTREME = {1'b0, {8{!S}}, 7'b0};
wire [15:0] ONE = 16'h3f80;

assign y = is_big ? EXTREME : (is_small ? ONE : approx);
endmodule