CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: vlfisher

vlfisher: fisher.cpp
	$(CXX) $(CFLAGS) fisher.cpp -o vlfisher -I ../../ -I /home/cy/lib/vlfeat -I /usr/local/include -L /home/cy/lib/vlfeat/bin/glnxa64 -l vl -lstdc++ -l z -l m -l opencv_highgui -l opencv_core
clean:
	rm -f *~ vlfisher
