#!/bin/bash
make
cd vl_gmm
gcc -o vlgmm vlgmm.cpp -I /home/cy/lib/vlfeat -L /home/cy/lib/vlfeat/bin/glnxa64 -l vl -lstdc++ 
cp vlgmm ../debug/ && cd ..
./debug/Main

