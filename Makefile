
CXX=clang++

all: reports/Exp.timing.rpt

src/Exp_LUT.v:
	mkdir -p build
	$(CXX) src/generate_direct_lut.cc -o build/generate_direct_lut
	cd src
	../build/generate_direct_lut

reports/%.timing.rpt: src/%.v
	echo "set toplevel $(basename $(notdir $^))" > prop.tcl
	genus -batch -files scripts/synth.tcl 

clean:
	@rm -rf reports/* genus*

