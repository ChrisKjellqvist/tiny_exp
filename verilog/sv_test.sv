module test();
logic q;
reg clk;
initial begin
	clk = 0;
end

forever begin
	#1;
	clk = ~clk;
end
always_ff @(posedge clk) begin
	q = !clk;
end
endmodule 
		
