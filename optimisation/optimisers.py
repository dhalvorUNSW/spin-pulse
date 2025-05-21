import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os
import platform
import pulses
import json

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
            ctypes.c_double,                          # det_max 
            ctypes.c_double,                          # amp_max
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
                      pulse_length,
                      n_max,
                      band_dig, 
                      amp_dig,
                      amp_max,
                      det_max,
                      init_temp,
                      cooling_rate,
                      w1_max,
                      lambda_val,
                      tau, save_pulse=False):
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

        # Save pulse as instance of ShapedPulse class
        times = np.linspace(0, tau, pulse_length)
        amps = self.coeffs_to_pulse(best_sin_coeffs, best_cos_coeffs, times, tau)
        shaped_pulse = pulses.ShapedPulse(amps, tau, det_max, amp_max, best_error)

        # Save pulse details to JSON
        if save_pulse == True:
            # Directory and filename
            directory = "PulseLibrary"
            filename = "custom_pulse.json"
            filepath = os.path.join(directory, filename)

            # Create the directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)

            # Write dictionary to a JSON file inside the directory
            with open(filepath, "w") as json_file:
                json.dump(shaped_pulse.encode_pulse(), json_file, indent=4)
        
        return shaped_pulse
    
    def coeffs_to_pulse(self, sin_coeffs, cos_coeffs, t, tau):  

        w = 2*np.pi/tau
        w1 = cos_coeffs[0]
    
        for i in range(len(sin_coeffs)):
            w1 += cos_coeffs[i+1]*np.cos((i+1)*w*t)
            w1 += sin_coeffs[i]*np.sin((i+1)*w*t)
    
        return w1 * w