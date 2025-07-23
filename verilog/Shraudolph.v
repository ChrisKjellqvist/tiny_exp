module Shraudolph(
    input clk,
    input rst,
    input [15:0] x,
    output [15:0] y
);

wire [31:0] fp32_out;
FP_FMA_FP32 fma(
    .clk(clk),
    .rst(rst),
    .in_mul0({x, 16'h0}),
    .in_mul1(32'h4b38aa3b),
    .in_add(32'h4e7de9a3),
    .out(fp32_out)
);

reg [31:0] fp32_stage;
wire [31:0] fp32_out;
FP_INTCAST_FP32 intcast(
    .clk(clk),
    .rst(rst),
    .in(fp32_stage),
    .out(fp32_out)
);

assign y = fp32_out[31:16];

endmodule