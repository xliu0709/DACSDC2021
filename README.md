# DACSDC2021

## CSIM test
1. run "make genweight" generate new weight file
2. run "make test" to compile test
3. run "./test" to generate output feature for the ultranet conv and dsp6 conv fun
4. run "diff conv_ultranet_out.txt conv_dsp6_out.txt" to check correctness

## SYN single conv module
1. run "vivado_hls hw_script/hls_single.tcl" with vivado_hls 2019.2  
