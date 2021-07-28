ARGS=$(getopt -l nobrick -o n -- "$@")
if [ "$?" != "0" ]; then
    exit 1
fi

if [[ "$ARGS" =~ '-n' ]]; then
    BARG="-DNO_BRICK"
else
    BARG=""
fi


hipcc laplacian-stencils-out.hip.cpp -DTILE=$1 -DPADDING=$2 $BARG -I ../bricklib/include -L ../bricklib/build/src -l brickhelper -O2 -fopenmp -o stencils
