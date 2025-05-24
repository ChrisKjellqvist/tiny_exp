
all: reports/Exp.rpt

reports/%.rpt: src/%.v
	echo "set toplevel %" > prop.tcl
	genus -batch scripts/synth.tcl 

