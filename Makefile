
CXX=clang++
CXXFLAGS=-std=c++20

all: reports/Exp_LUT.timing.rpt reports/Exp.timing.rpt

src/Exp_LUT.v: src/generate_direct_lut.cc
	mkdir -p build
	$(CXX) $(CXXFLAGS) src/generate_direct_lut.cc -o build/generate_direct_lut
	cd src && ../build/generate_direct_lut

reports/%.timing.rpt: src/%.v
	echo "set toplevel $(basename $(notdir $^))" > prop.tcl
	genus -batch -files scripts/synth.tcl 

clean:
	@rm -rf reports/* genus* src/Exp_LUT.v fv build prop.tcl
