`timescale 1ns/1ps

module TestBench();

reg [15:0] in;
wire [15:0] out;
HardCodedMAC_BF16 dut(
    .clk(1'b0),
    .x(in),
    .y(out)
);

integer i;
initial begin
    $dumpfile("dump.vcd");
    $dumpvars(0);
    for (i = 0; i < 1024; i=i+1) begin
        in = 16'h3f80 + i;
        $display("In: 0x%04x\tOut: 0x%04x", in, out);
        #1;
    end
end

endmodule
