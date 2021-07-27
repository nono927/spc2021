#!/bin/bash
#PJM -L rscgrp=lecture-o
#PJM -L node=12
#PJM --mpi proc=288
#PJM --omp thread=2
#PJM -L elapse=00:15:00
#PJM -g gt68

module load fj fjmpi

mpiexec -stdout  spc.out ./spc

# fipp -C -d ./profile_dir -Icall -i10 -Sregion mpiexec -stdout  spc.out ./spc
# fipppx -A -d ./profile_dir -Icpupa,balance,call,src,mpi > fipp.txt

# export FLIB_FASTOMP=TRUE
# fapp -C -d ./profile_dir -Icpupa -Hevent=statistics mpiexec -stdout  spc.out ./spc
# fapppx -A -d ./profile_dir -Icpupa,mpi -ttext -o fipp.txt
