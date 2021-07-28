#!/bin/bash

#SBATCH -A CSC383
#SBATCH -t 00:10:00
#SBATCH -p ecp
#SBATCH -N 1

./codegen.sh 8 32 $@ && \
./compile.sh 8 32 $@ && \
# srun -A CSC383 -t 15 -p ecp -N 1 
rocprof --timestamp on --stats -i config.txt -o ./profs/config-8-32.csv stencils
echo 8 32
python stats_process.py ./profs/config-8-32.csv
