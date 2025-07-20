module TestBench();

reg in_valid;
reg [15:0] in_data;
wire in_ready;

reg out_ready;
wire [15:0] out_data;
wire out_valid;

reg clk;
reg rst;

wire [15:0] out;
cheby_BF16 dut(
    .clk(clk),
    .rst(rst),
    .in_valid(in_valid),
    .in_data(in_data),
    .in_ready(in_ready),
    .out_ready(out_ready),
    .out_data(out_data),
    .out_valid(out_valid)
);

integer i;
initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0);
    clk = 0;
    rst = 1;
    out_ready = 0;
    in_data = 0;
    in_valid = 0;

    for (i=0;i<10;i=i+1) begin
        #1 clk = !clk;
    end
    rst = 0;
    for (i=0;i<10;i=i+1) begin
        #1 clk = !clk;
    end

    in_valid = 1;
    in_data = 16'h3f80;
    while (in_ready == 0) begin
        i = i + 1;
        #1 clk = ~clk;
        #1 clk = ~clk;
        $display("Waited %d cycles for in_ready", i);
	if (i>1000) begin
		$finish();
	end
    end
    #1 clk = ~clk;
    #1 clk = ~clk;
    in_valid = 0;

    #1 clk = ~clk;
    #1 clk = ~clk;
    out_ready = 1;
    for (i=0;i<10;i=i+1) begin
        if (out_valid) begin
            $display("Received %04x", out_data);
            $finish();
        end
        #1 clk = ~clk;
        #1 clk = ~clk;
    end
end

endmodule
