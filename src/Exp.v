
module Exp (
	input clk,
	input [15:0] data_i,
	output [15:0] data_o);
reg [15:0] in_flop;
reg [15:0] out_flop;
wire data_s = in_flop[15];
wire [7:0] data_e = in_flop[14:7];
wire [6:0] data_m = in_flop[6:0];

reg [15:0] base;
reg [15:0] offset;
reg [22:0] offset_mul;
wire hi = data_e > 131;
wire lo = data_e <= 122;
assign data_o = out_flop;
always @(posedge clk) begin
	in_flop <= data_i;
	if (hi) begin
	    if (data_s) begin
		    out_flop <= 16'h7f80;
		end
	end else if (lo) begin
		out_flop <= 16'38f0;
	end else begin
		case (data_e)
		    121: begin
		        base = 16'h3f82;
		        offset = 2;
		    end
		    122: begin
		        base = 16'h3f84;
		        offset = 4;
            end
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
			132: begin
			    base = 16'h568f;
			    offset = 5906;
            end
            133: begin
                base = 16'h6da1;
                offset = 11817;
            end
		endcase
		offset_mul = data_m * offset;
		out_flop <= base + offset_mul[22:7];
	end
end
endmodule

