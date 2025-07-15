.PHONY: all lut reg sram
all: lut reg sram
	@echo "Lint pass"

lut:
	@verilator --lint-only \
		HardCodedMAC_BF16.v

reg:
	@verilator --lint-only \
		RegFileMAC_BF16.v

sram:
	@verilator --lint-only \
		-Wno-USERWARN -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-UNOPTFLAT -Wno-ASCRANGE \
		-Wno-UNSIGNED \
		-I./cvfpu/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ \
		-I./cvfpu/openc910/C910_RTL_FACTORY/gen_rtl/clk/rtl/ \
		-Icvfpu \
		-Icvfpu/fpu_div_sqrt_mvp/hdl \
		-Icvfpu/common_cells/include \
		-Icvfpu/common_cells/src \
		cvfpu/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv \
		cvfpu/common_cells/src/cf_math_pkg.sv \
		cvfpu/fpnew_pkg.sv \
		SRAMBaseline_BF16.v

sram_32:
	@verilator --lint-only \
		-Wno-USERWARN -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-UNOPTFLAT -Wno-ASCRANGE \
		-Wno-UNSIGNED \
		-I./cvfpu/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ \
		-I./cvfpu/openc910/C910_RTL_FACTORY/gen_rtl/clk/rtl/ \
		-Icvfpu \
		-Icvfpu/fpu_div_sqrt_mvp/hdl \
		-Icvfpu/common_cells/include \
		-Icvfpu/common_cells/src \
		cvfpu/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv \
		cvfpu/common_cells/src/cf_math_pkg.sv \
		cvfpu/fpnew_pkg.sv \
		SRAMBaseline_FP32.v
