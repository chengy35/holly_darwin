#!/bin/bash
make
cd vl_gmm
gcc -o vlgmm vlgmm.cpp -I /home/cy/lib/vlfeat -L /home/cy/lib/vlfeat/bin/glnxa64 -l vl -lstdc++ 
cp vlgmm ../debug/ && cd ..

cd vl_fisher
gcc -o vlfisher fisher.cpp -I /home/cy/lib/vlfeat -I /usr/local/include -L /home/cy/lib/vlfeat/bin/glnxa64 -l vl -lstdc++ -l z -l m -l opencv_highgui -l opencv_core
cd ..
cp ./vl_fisher/vlfisher ./debug/
./debug/Main

