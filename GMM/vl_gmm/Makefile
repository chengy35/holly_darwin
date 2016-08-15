CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: vlgmm

vlgmm: vlgmm.cpp
	$(CXX) $(CFLAGS) vlgmm.cpp -o vlgmm -I /home/cy/lib/vlfeat -L /home/cy/lib/vlfeat/bin/glnxa64 -l vl -lstdc++
clean:
	rm -f *~ vlgmm
