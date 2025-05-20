import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os
import platform

class SimulatedAnnealing:
    """Python wrapper for Fortran simulated annealing library"""
    
    def __init__(self, library_path=None):
        """
        Initialize the SimulatedAnnealing class.
        
        Parameters:
        -----------
        library_path : str, optional
            Path to the compiled Fortran shared library.
            If None, will look in the current directory.
        """
        # Determine the library file extension based on the platform
        if platform.system() == "Windows":
            lib_ext = ".dll"
        elif platform.system() == "Darwin":  # macOS
            lib_ext = ".dylib"
        else:  # Linux and others
            lib_ext = ".so"
            
        # Set default library path if not provided
        if library_path is None:
            library_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                       f"annealing_lib{lib_ext}")
        
        # Load the library
        self.lib = ctypes.CDLL(library_path)
        
        # Define the argument types for the run_annealing function
        self.lib.run_annealing.argtypes = [
            ctypes.c_int,                             # Np
            ctypes.c_int,                             # n_max
            ctypes.c_int,                             # band_dig
            ctypes.c_int,                             # amp_dig
            ctypes.c_double,                          # amp_max
            ctypes.c_double,                          # det_max 
            ctypes.c_double,                          # init_temp
            ctypes.c_double,                          # cooling_rate
            ctypes.c_double,                          # w1_max
            ctypes.c_double,                          # lambda
            ctypes.c_double,                          # tau
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # best_sin_coeffs (output)
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # best_cos_coeffs (output)
            ctypes.POINTER(ctypes.c_double)           # best_error (output)
        ]
        self.lib.run_annealing.restype = None  # This function doesn't return a value

    def run_annealing(self, 
                      n_max=20,
                      pulse_length=100,
                      tau=100.0e-9,
                      band_dig=6, 
                      amp_dig=5,
                      det_max=0.1/100.0e-9,
                      amp_max=0.05,
                      init_temp=5.0,
                      cooling_rate=0.95,
                      w1_max=2.0 * np.pi * 80.0e6,
                      lambda_val=1.0e3):
        """
        Run the simulated annealing algorithm.
        
        Arguments:
        -----------
        n_max : int
            Maximum number of Fourier coefficients
        pulse_length : int
            Length of the pulse array
        band_dig : int
            Samples per 1/tau in spectrum
        amp_dig : int
            Digits in amplitude
        init_temp : float
            Initial temperature
        cooling_rate : float
            Temperature cooling rate
        w1_max : float
            Maximum pulse amplitude
        lambda_val : float
            Penalty parameter
        amp_max : float
            Maximum amplitude
        det_max : float
            Maximum detuning
        tau : float
            Time constant
        init_coeffs : numpy.ndarray, optional
            Initial Fourier coefficients. If None, will use [0.25, 0, 0, ...]
            
        Returns:
        --------
        tuple
            (best_sin_coeffs, best_cos_coeffs, best_error) where:
            - best_coeffs: numpy.ndarray of optimized Fourier coefficients
            - best_error: float with the best error value achieved
        """
        # Create output array for best coefficients
        best_sin_coeffs = np.zeros(n_max, dtype=np.float64)
        best_cos_coeffs = np.zeros(n_max + 1, dtype=np.float64)
        
        # Create a C double pointer for the best error output
        best_error_c = ctypes.c_double(0.0)
                
        # Call the Fortran function
        self.lib.run_annealing(
            pulse_length, n_max, band_dig, amp_dig, amp_max, det_max, 
            init_temp, cooling_rate, w1_max, lambda_val, tau, 
            best_sin_coeffs, best_cos_coeffs, ctypes.byref(best_error_c)
        )
        
        # Convert C double to Python float
        best_error = best_error_c.value
        
        return best_sin_coeffs, best_cos_coeffs, best_error