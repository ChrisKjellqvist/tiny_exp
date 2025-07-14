module RegFileMAC_BF16 (
    input         clk,
    input  [15:0] x,
    output [15:0] y,

    input         cfg_w_en,
    input         cfg_sgn,
    input [3:0]   cfg_idx,
    input [15:0]  cfg_base,
    input [25:0]  cfg_offset
);

reg [15:0] BASES   [0:1][0:12];
reg [25:0] OFFSETS [0:1][0:12];

always @(posedge clk) begin
    if (cfg_w_en) begin
        BASES[cfg_sgn][cfg_idx] <= cfg_base;
        OFFSETS[cfg_sgn][cfg_idx] <= cfg_offset;
    end
end

wire       S = x[15];
wire [7:0] E = x[14:7];
wire [6:0] M = x[6:0];

localparam Emin = -7;
localparam Emax = 6;

wire [7:0] E_norm = E - (127+Emin);
wire [3:0] E_adj = E_norm[3:0];

wire [15:0] base = BASES[S][E_adj];
wire [25:0] offset = OFFSETS[S][E_adj];

wire [25:0] product = M * offset;
wire [15:0] product_used = product[22:7];

wire [15:0] approx = base + product_used;

wire is_big = E >= (127 + Emax);
wire is_small = E < (127 + Emin);

wire [15:0] EXTREME = {1'b0, {8{!S}}, 7'b0};
wire [15:0] ONE = 16'h3f80;

assign y = is_big ? EXTREME : (is_small ? ONE : approx);
endmodule
