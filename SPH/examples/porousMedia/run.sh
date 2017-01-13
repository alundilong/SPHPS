#! /bin/bash

set -e
set -u

lmp=../../src/lmp_mpi
np=4

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
mpirun -np ${np} ${lmp} -in  in.porous
