module cheby_BF16(
    input clk,
    input rst,
    input in_valid,
    output in_ready,
    input [15:0] in_data,
    output out_valid,
    input out_ready,
    output [15:0] out_data
)
`include "cheby_poly.v"

localparam S_IDLE = 0;
localparam S_WORK = 1;
localparam S_REPLY = 2;

reg [1:0] state;
assign in_ready = state == S_IDLE;
assign out_valid = state == S_REPLY;

reg [15:0] sq_acc;
reg [15:0] res_acc;
assign out_date = res_acc;
reg [15:0] x_reg;
reg [15:0] x_int_reg;
reg [1:0] poly_count;

wire [15:0] casted_x;
wire [15:0] x_int_adj = 64 + casted_x;

FP_INTCAST_BF16 cast(
    .clk(clk),
    .rst(rst),
    .inA(in_data),
    .out(casted_x)
);

wire [15:0] res_update;
FP_FMA_BF16 fma(
    .clk(clk),
    .rst(rst),
    .in_mul0(constants[poly_count][x_int_reg]),
    .in_mul1(sq_acc),
    .in_add(res_update)
);

wire [15:0] mul_result;
FP_MUL_FP32 mul(
    .clk(clk),
    .rst(rst),
    .inA(sq_acc),
    .inB(x_reg),
    .out(mul_result)
);

always @(posedge clk) begin
    if (rst) begin
        state <= S_IDLE;
    end else begin
        if (state == S_IDLE) begin
            if (in_valid) begin
                x_reg <= in_data;
                x_int_reg <= x_int_adj;
                state <= S_WORK;
                sq_acc <= 1;
                poly_count <= 0;
            end
        end else if (state == S_WORK) begin
            poly_count <= poly_count + 1;
            res_acc <= res_update;
            sq_acc <= mul_result;
            if (poly_count == 3) begin
                state <= S_REPLY;
            end
        end else if (state == S_REPLY) begin
            if (out_ready) begin
                state <= S_IDLE
            end
        end
    end
end

endmodule
