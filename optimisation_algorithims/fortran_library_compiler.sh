#!/bin/bash
# Script to compile the Fortran modules as a shared library

# Create build directory if it doesn't exist
mkdir -p build

# Compile all modules with position-independent code
gfortran -c -fPIC -O3 sub/matrix_utils.f90 -o build/matrix_utils.o
gfortran -c -fPIC -O3 sub/coeffs_to_pulse.f90 -o build/coeffs_to_pulse.o
gfortran -c -fPIC -O3 sub/update_pulse.f90 -o build/update_pulse.o
gfortran -c -fPIC -O3 sub/sim_functions.f90 -o build/sim_functions.o
gfortran -c -fPIC -O3 sub/cost_functions.f90 -o build/cost_functions.o
gfortran -c -fPIC -O3 py_annealing_optimiser.f90 -o build/py_annealing.o

# Create shared library
gfortran -shared -o libannealing.so build/*.o

echo "Library compilation complete. The shared library is libannealing.so"