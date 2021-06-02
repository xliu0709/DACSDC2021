############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project HLS
set_top C
add_files src/single_test_conv1x1.cpp -cflags "-std=c++11" -csimflags "-std=c++11"
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 10 -name default
#source "./HLS_SINGLE/solution1/directives.tcl"
#csim_design
csynth_design
# export_design -flow syn -rtl verilog -format ip_catalog
#cosim_design
export_design -format ip_catalog
