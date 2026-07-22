#!/bin/bash
set -e

echo "=== Installing MPI ==="
julia --project=~/projects/ddp_flux/test -e '
import Pkg
Pkg.add("MPI")
' 2>&1

echo "=== Running reduce_distributedtest ==="
~/.julia/bin/mpiexecjl --project=~/projects/ddp_flux/test -n 2 julia ~/projects/ddp_flux/test/ext_distributed/reduce_distributedtest.jl mpi
echo "=== REDUCE PASSED ==="

echo "=== Running unused_parameters_distributedtest ==="
~/.julia/bin/mpiexecjl --project=~/projects/ddp_flux/test -n 2 julia ~/projects/ddp_flux/test/ext_distributed/unused_parameters_distributedtest.jl mpi
echo "=== UNUSED_PARAMS PASSED ==="

echo "=== ALL TESTS PASSED ==="
