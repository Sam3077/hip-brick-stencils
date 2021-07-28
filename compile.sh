hipcc laplacian-stencils-out.hip.cpp -DTILE=$1 -DPADDING=$2 -I ../bricklib/include -L ../bricklib/build/src -l brickhelper -O2 -fopenmp -o stencils
