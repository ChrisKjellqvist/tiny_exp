
all: reports/Exp.rpt

reports/%.rpt: src/%.v
	echo "set toplevel $(basename $(notdir $^))" > prop.tcl
	genus -batch -files scripts/synth.tcl 

clean:
	@rm -rf reports/*

