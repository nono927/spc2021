#!/bin/bash
#PJM -L rscgrp=lecture-o
#PJM -L node=12
#PJM --mpi proc=576
#PJM -L elapse=00:15:00
#PJM -g gt68

module load fj fjmpi

mpiexec -stdout  spc.out ./spc
# fipp -C -d ./profile_dir -Icall -i10 mpiexec -stdout  spc.out ./spc
# fipppx -A -d ./profile_dir -Icpupa,balance,call,src,mpi > fipp.txt
