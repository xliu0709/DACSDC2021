############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project HLS_SINGLE
set_top conv3x3_bn_act_DSPopt_hls_wrapper
add_files src/single_test.cpp -cflags "-std=c++11" -csimflags "-std=c++11"
open_solution "solution1"
set_part {xczu3eg-sbva484-1-e}
create_clock -period 4 -name default
#source "./HLS_SINGLE/solution1/directives.tcl"
#csim_design
csynth_design
#cosim_design
# export_design -format ip_catalog