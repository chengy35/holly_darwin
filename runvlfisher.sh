#!/bin/bash
make
cd vl_fisher
gcc -o vlfisher fisher.cpp -I /home/cy/lib/vlfeat -I /usr/local/include -L /home/cy/lib/vlfeat/bin/glnxa64 -l vl -lstdc++ -l z -l m -l opencv_highgui -l opencv_core
cd ..
cp ./vl_fisher/vlfisher ./debug/
./debug/Main
