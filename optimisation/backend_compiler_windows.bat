@echo off
REM Script to compile the Fortran modules as a shared library for Windows

REM Create build directory if it doesn't exist
if not exist build mkdir build

REM Compile all modules with position-independent code
echo Compiling Fortran modules...
gfortran -c -O3 sub/matrix_utils.f90 -o build/matrix_utils.o
if %errorlevel% neq 0 goto :error

gfortran -c -O3 sub/coeffs_to_pulse.f90 -o build/coeffs_to_pulse.o
if %errorlevel% neq 0 goto :error

gfortran -c -O3 sub/update_pulse.f90 -o build/update_pulse.o
if %errorlevel% neq 0 goto :error

gfortran -c -O3 sub/sim_functions.f90 -o build/sim_functions.o
if %errorlevel% neq 0 goto :error

gfortran -c -O3 sub/cost_functions.f90 -o build/cost_functions.o
if %errorlevel% neq 0 goto :error

REM gfortran -c -O3 py_annealing_optimiser.f90 -o build/py_annealing.o
REM if %errorlevel% neq 0 goto :error

gfortran -c -O3 annealing_algorithm_backend.f90 -o build/annealing_algorithm_backend.o
if %errorlevel% neq 0 goto :error

REM Create shared library (DLL for Windows)
echo Creating shared library...
gfortran -shared -o annealing_lib.dll build/*.o
if %errorlevel% neq 0 goto :error

echo Library compilation complete. The shared library is annealing_lib.dll
goto :end

:error
echo Compilation failed with error code %errorlevel%
pause
exit /b %errorlevel%

:end
pause