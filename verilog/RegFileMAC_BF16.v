module RegFileMAC_BF16 (
    input         clk,
    input  [15:0] x,
    output [15:0] y,

    input         cfg_w_en,
    input         cfg_sgn,
    input [3:0]   cfg_idx,
    input [15:0]  cfg_base,
    input [15:0]  cfg_offset
);

reg [15:0] BASES   [0:1][0:12];
reg [15:0] OFFSETS [0:1][0:12];

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

wire [7:0] E_adj = E + (127+Emin);

wire [15:0] base = BASES[S][E_adj];
wire [15:0] offset = OFFSETS[S][E_adj];

wire [22:0] product = M * offset;
wire [15:0] product_used = product[22:7];

wire [15:0] approx = base + product_used;

wire is_big = E >= (127 + Emax);
wire is_small = E < (127 + Emin);

wire [15:0] INF_PLUS = 16'h7f80;
wire [15:0] INF_MINUS = 16'hff80;

assign y = is_big ? (S ? INF_MINUS : INF_PLUS) : (is_small ? 0 : approx);
endmodule
