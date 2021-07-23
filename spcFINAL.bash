#!/bin/bash
#PJM -L rscgrp=lecture-o
#PJM -L node=12
#PJM --mpi proc=576
#PJM -L elapse=00:15:00
#PJM -g gt68

module load fj fjmpi

cp spcDEBUG1.h spc.h 
make -f Makefile.native clean
make -f Makefile.native
mpiexec -stdout spcFINAL1.out ./spc
cp spcDEBUG2.h spc.h 
make -f Makefile.native clean
make -f Makefile.native
mpiexec -stdout spcFINAL2.out ./spc
cp spcFINAL1.h spc.h 
make -f Makefile.native clean
make -f Makefile.native
mpiexec -stdout spcFINAL3.out ./spc
cp spcFINAL2.h spc.h 
make -f Makefile.native clean
make -f Makefile.native
mpiexec -stdout spcFINAL4.out ./spc

