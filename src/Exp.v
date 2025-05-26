
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
		out_flop <= 16'h38f0;
	end else begin
		case (data_e)
			121: begin
				if (!data_s) begin
					base = 16'h3f82;
					offset = 2;
				end else begin
					base = 16'h3f7c;
					offset = 4;
				end
			end
			122: begin
				if (!data_s) begin
					base = 16'h3f84;
					offset = 4;
				end else begin
					base = 16'h3f78;
					offset = 8;
				end
			end
			123: begin
				if (!data_s) begin
					base = 16'h3F88;
					offset = 9;
				end else begin
					base = 15'h3f70;
					offset = 15;
				end
			end
			124: begin
				if (!data_s) begin
					base = 16'h3f91;
					offset = 19;
				end else begin
					base = 16'h3f61;
					offset = 26;
				end
			end
			125: begin
				if (!data_s) begin
					base = 16'h3fa4;
					offset = 47;
				end else begin
					base = 16'h3f47;
					offset = 44;
				end
			end
			126: begin
				if (!data_s) begin
					base = 16'h3fd3;
					offset = 90;
				end else begin
					base = 16'h3f1b;
					offset = 95;
				end
			end
			127: begin
				if (!data_s) begin
					base = 16'h402d;
					offset = 191;
				end else begin
					base = 16'h3ebc;
					offset = 178;
				end
			end
			128: begin
				if (!data_s) begin
					base = 16'h40ec;
					offset = 366;
				end else begin
					base = 16'h3e0a;
					offset = 372;
				end
			end
			129: begin
				if (!data_s) begin
					base = 16'h425a;
					offset = 736;
				end else begin
					base = 16'h3c96;
					offset = 743;
				end
			end
			130: begin
				if (!data_s) begin
					base = 16'h453a;
					offset = 1485;
				end else begin
					base = 16'h39af;
					offset = 1470;
				end
			end
			131: begin
				if (!data_s) begin
					base = 16'h4b07;
					offset = 2952;
				end else begin
					base = 16'h33f1;
					offset = 2957;
				end
			end
			132: begin
				if (!data_s) begin
					base = 16'h568f;
					offset = 5906;
				end else begin
					base = 16'h2864;
					offset = 5913;
				end
			end
			133: begin
				if (!data_s) begin
					base = 16'h6da1;
					offset = 11817;
				end else begin
					base = 16'h114b;
					offset = 11818;
				end
			end
		endcase
		offset_mul = data_m * offset;
		out_flop <= data_s ? base - offset_mul[22:7] : base + offset_mul[22:7];
	end
end
endmodule

