
ARGS=$(getopt -l nobrick -o n -- "$@")
if [ "$?" != "0" ]; then
    exit 1
fi

if [[ "$ARGS" =~ '-n' ]]; then
    BARG="-DNO_BRICK"
else
    BARG=""
fi

python ../bricklib/codegen/vecscatter laplacian-stencils.hip.cpp laplacian-stencils-out.hip.cpp -c cpp -- \
-DTILE=$1 -DPADDING=$2 $BARG -fopenmp -O2 -I../bricklib/include -D__HIP_PLATFORM_HCC__= -I/sw/spock/spack-envs/views/rocm-4.1.0/hip/include/ \
-I/sw/spock/spack-envs/views/rocm-4.1.0/hip/include -I/sw/spock/spack-envs/views/rocm-4.1.0/llvm/bin/../lib/clang/12.0.0 \
-I/sw/spock/spack-envs/views/rocm-4.1.0/include -D__HIP_ROCclr__
