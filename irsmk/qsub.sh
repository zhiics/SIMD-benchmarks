#!/bin/bash
#$ -N massif_irsmk_dbl
#$ -q free64
#$ -m eas

# Module load Cuda Compilers and GCC
#module load  cuda/5.0
#module load  gcc/4.4.3

# Copy the cuda program and necessary cuda include files:
#cp ~demo/cuda/* .

make

#valgrind --tool=cachegrind ./IRSmk
valgrind --tool=massif ./IRSmk
