module cheby_BF16_piped(
    input clk,
    input rst,
    input [15:0] in_data,
    output [15:0] out_data
);
`include "cheby_poly.v"

reg [15:0] sq_shreg [0:1]; // for x^2 and x^3
reg [15:0] res_acc [0:3]; // for T1, T2, T3 (not T0)
assign out_date = res_acc[2];
reg [15:0] x_shreg [0:3];
reg [15:0] x_int_shreg [0:3];

wire [15:0] chosen_constants[0:3];
assign chosen_constants[0] = constants[0][x_int_shreg[0]];
assign chosen_constants[1] = constants[1][x_int_shreg[1]];
assign chosen_constants[2] = constants[2][x_int_shreg[2]];
assign chosen_constants[3] = constants[3][x_int_shreg[3]];

wire [15:0] casted_x;
wire [15:0] x_int_adj = 64 + casted_x;

FP_INTCAST_BF16 cast(
    .clk(clk),
    .rst(rst),
    .inA(in_data),
    .out(casted_x)
);

wire [15:0] sq1_res;
FP_MUL_BF16 sq1(
    .clk(clk),
    .rst(rst),
    .inA(x_shreg[1]),
    .inB(x_shreg[1]),
    .out(sq1_res)
);
wire [15:0] sq2_res;
FP_MUL_BF16 sq2(
    .clk(clk),
    .rst(rst),
    .inA(sq_shreg[0]),
    .inB(x_shreg[2]),
    .out(sq2_res)
);

wire [15:0] fma1_res;
FP_FMA_BF16 fma1(
    .clk(clk),
    .rst(rst),
    .in_mul0(chosen_constants[1]),
    .in_mul1(x_shreg[1]),
    .in_add(res_acc[0]),
    .out(fma1_res)
);

wire [15:0] fma2_res;
FP_FMA_BF16 fma2(
    .clk(clk),
    .rst(rst),
    .in_mul0(chosen_constants[2]),
    .in_mul1(sq_shreg[0]),
    .in_add(res_acc[1]),
    .out(fma2_res)
);

wire [15:0] fma3_res;
FP_FMA_BF16 fma3(
    .clk(clk),
    .rst(rst),
    .in_mul0(chosen_constants[3]),
    .in_mul1(sq_shreg[1]),
    .in_add(res_acc[2]),
    .out(fma3_res)
);

assign out_data = res_acc[3];


always @(posedge clk) begin
    x_shreg[0] <= in_data;
    x_shreg[1] <= x_shreg[0];
    x_shreg[2] <= x_shreg[1];
    x_shreg[3] <= x_shreg[2];
    x_int_shreg[0] <= x_int_adj;
    x_int_shreg[1] <= x_int_shreg[0];
    x_int_shreg[2] <= x_int_shreg[1];
    x_int_shreg[3] <= x_int_shreg[2];
    sq_shreg[0] <= sq1_res;
    sq_shreg[1] <= sq2_res;
    res_acc[0] <= chosen_constants[0];
    res_acc[1] <= fma1_res;
    res_acc[2] <= fma2_res;
    res_acc[3] <= fma3_res;
end

endmodule
