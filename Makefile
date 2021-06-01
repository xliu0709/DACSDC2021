
# CINCLUDES	+= -I../../finn-hlslib/
CINCLUDES 	+= -I /opt/Xilinx/Vivado/2019.1/include
# CINCLUDES 	+= -std=c++0x -Wall -Wno-unknown-pragmas -Wall -Wno-unknown-pragmas -Wno-unused-variable -g
CXX			= g++
# LDFLAGS		= -lpthread
CFLAGS		+= -std=c++11
CFLAGS 		+= -fsanitize=address
CFLAGS 		+= -g
# CFLAGS		+= -O3

genweight:
	$(CXX) $(CFLAGS) $(CINCLUDES) src/gen_weight.cpp -o genweight 

test: 
	$(CXX) $(CFLAGS) $(CINCLUDES) src/ultranet.cpp -o test 

clean:
	rm -f test
	rm -f genweight