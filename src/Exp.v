
module Exp (
  input clk,
  input rst,
  input [15:0] data_i,
  output [15:0] data_o);

wire data_s = data_i[15];
wire data_e = data_i[14:7];
wire data_m = data_i[6:0];

reg [15:0] base;
reg [15:0] offset;
// exponents in FP are really E+127, so e=0 is really E=127
always @(posedge clk) begin
  if (data_e <= 122) begin
    data_o = 0;
  end else if (data_e > 131) begin
    data_o = 16'h7F10;
  end else begin
    case (data_e) begin
      123: begin
        base = 16'h3F88;
        offset = 9;
      end
      124: begin
        base = 16'h3f91;
        offset = 19;
      end
      125: begin
        base = 16'h3fa4;
        offset = 47;
      end
      126: begin
        base = 16'h3fd3;
        offset = 90;
      end
      127: begin
        base = 16'h402d;
        offset = 191;
      end
      128: begin
        base = 16'h40ec;
        offset = 366;
      end
      129: begin
        base = 16'h425a;
        offset = 736;
      end
      130: begin
        base = 16'h453a;
        offset = 1485;
      end
      131: begin
        base = 16'h4b07;
        offset = 2952;
      end
    endcase
  end
end
endmodule

