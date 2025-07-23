FLAGS_FPU= -y ./cvfpu/openc910/C910_RTL_FACTORY/gen_rtl/vfdsu/rtl/ \
		-y ./cvfpu/openc910/C910_RTL_FACTORY/gen_rtl/clk/rtl/ \
		-y cvfpu \
		-y cvfpu/fpu_div_sqrt_mvp/hdl \
		+incdir+cvfpu/common_cells/include \
		-y cvfpu/common_cells/include \
		-y cvfpu/common_cells/src \
		+incdir+cvfpu/common_cells \
		-y ./ \
		-full64 \
		-sverilog \
		cvfpu/fpu_div_sqrt_mvp/hdl/defs_div_sqrt_mvp.sv \
		cvfpu/common_cells/src/cf_math_pkg.sv \
		cvfpu/fpnew_pkg.sv \
		./cvfpu/common_cells/src/lzc.sv \
		cvfpu/common_cells/src/rr_arb_tree.sv \
		./cvfpu/fpnew_cast_multi.sv\
		./cvfpu/fpnew_classifier.sv\
		./cvfpu/fpnew_divsqrt_multi.sv\
		./cvfpu/fpnew_divsqrt_th_32.sv\
		./cvfpu/fpnew_divsqrt_th_64_multi.sv\
		./cvfpu/fpnew_fma.sv\
		./cvfpu/fpnew_fma_multi.sv\
		./cvfpu/fpnew_noncomp.sv\
		./cvfpu/fpnew_opgroup_block.sv\
		./cvfpu/fpnew_opgroup_fmt_slice.sv\
		./cvfpu/fpnew_opgroup_multifmt_slice.sv\
		./cvfpu/fpnew_rounding.sv\
		./cvfpu/fpnew_top.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/control_mvp.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/div_sqrt_mvp_wrapper.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/div_sqrt_top_mvp.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/iteration_div_sqrt_mvp.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/norm_div_sqrt_mvp.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/nrbd_nrsc_mvp.sv\
		./cvfpu/fpu_div_sqrt_mvp/hdl/preprocess_mvp.sv\


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
	verilator --lint-only \
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

sram_vcs_fp32:
	vcs $(FLAGS_FPU) \
		SRAMBaseline_FP32.v \
		./FP_MUL_FP32.sv \
		testbench_ram.v

sram_vcs_bf16:
	vcs $(FLAGS_FPU) \
		SRAMBaseline_BF16.v \
		./FP_MUL_BF16.sv \
		testbench_rambf16.v

cheby:
	vcs $(FLAGS_FPU) \
		./FP_MUL_BF16.sv \
		./FP_INTCAST_BF16.sv \
		./FP_FMA_BF16.sv \
		./cheby_BF16.v \
		cheby_tb.v
chebyp:
	vcs $(FLAGS_FPU) \
		./FP_MUL_BF16.sv \
		./FP_INTCAST_BF16.sv \
		./FP_FMA_BF16.sv \
		./cheby_BF16_piped.v \
		chebyp_tb.v
