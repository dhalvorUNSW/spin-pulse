import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer
import os
import platform
import pulses

class evolveState_fast():
    """ 
    Python wrapper for fortran function to evolve quantum state.
    """

    def __init__(self, library_path=None):
        """
        Initialise evolveState parameters:

        Args:
            - library_path (optional) : path to compiled fortran library.
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
                                       f"libs/backend_lib{lib_ext}")
            
        # Load the library
        self.lib = ctypes.CDLL(library_path)

        # Define argument types
        self.lib.evolveState_fast.argtypes = [
            ctypes.c_int,                    # Np
            ctypes.c_double,                 # dt
            ctypes.POINTER(ctypes.c_double), # w1x array
            ctypes.POINTER(ctypes.c_double), # w1y array
            ctypes.POINTER(ctypes.c_double), # det array
            ctypes.POINTER(ctypes.c_double), # psi0 real part
            ctypes.POINTER(ctypes.c_double), # psi0 imaginary part
            ctypes.POINTER(ctypes.c_double), # states output real part
            ctypes.POINTER(ctypes.c_double), # states output imaginary part
        ]

        # No return value (subroutine)
        self.lib.evolveState_fast.restype = None


    def evolve(self, dt, w1x, w1y, det, psi0):
        """
        Evolve quantum state using the Fortran backend.
        
        Args:
            dt (float): Time step
            w1x (array): Array of x-component control amplitudes
            w1y (array): Array of y-component control amplitudes  
            det (array): Array of detuning values
            psi0 (complex array): Initial state [psi0[0], psi0[1]]
            
        Returns:
            numpy.ndarray: Complex array of shape (2, Np) containing evolved states
        """
        # Convert inputs to numpy arrays
        w1x = np.asarray(w1x, dtype=np.float64)
        w1y = np.asarray(w1y, dtype=np.float64)
        det = np.asarray(det, dtype=np.float64)
        psi0 = np.asarray(psi0, dtype=np.complex128)
        
        # Check that all control arrays have the same length
        Np = len(w1x)
        if len(w1y) != Np or len(det) != Np:
            raise ValueError("All control arrays (w1x, w1y, det) must have the same length")
        
        # Check that psi0 has length 2
        if len(psi0) != 2:
            raise ValueError("Initial state psi0 must have length 2")
        
        # Prepare output arrays
        states_real = np.zeros((2, Np), dtype=np.float64, order='F')
        states_imag = np.zeros((2, Np), dtype=np.float64, order='F')
        
        # Extract real and imaginary parts of psi0
        psi0_real = np.array([psi0[0].real, psi0[1].real], dtype=np.float64)
        psi0_imag = np.array([psi0[0].imag, psi0[1].imag], dtype=np.float64)
        
        # Call the Fortran function
        self.lib.evolveState_fast(
            ctypes.c_int(Np),
            ctypes.c_double(dt),
            w1x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            w1y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            det.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            psi0_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            psi0_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            states_real.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            states_imag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        
        # Combine real and imaginary parts
        states = states_real.T + 1j * states_imag.T
        
        return states

class GradientAscent:

    def __init__(self, library_path=None):

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
                                       f"libs/backend_lib{lib_ext}")
        
        # Load the library
        self.lib = ctypes.CDLL(library_path)
        
        # Define the argument types for the run_annealing function
        self.lib.run_grad_ascent.argtypes = [
            ctypes.c_int,                             # Np
            ctypes.c_int,                             # band_dig
            ctypes.c_int,                             # amp_dig
            ctypes.c_double,                          # det_max 
            ctypes.c_double,                          # amp_max
            ctypes.c_double,                          # w1_max
            ctypes.c_double,                          # learning rate
            ctypes.c_double,                          # tau
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # w1x (output)
            ctypes.POINTER(ctypes.c_double)           # best_error (output)
        ]
        self.lib.run_grad_ascent.restype = None  # This function doesn't return a value

    def run_grad_ascent(self,
                        Np,
                        band_dig,
                        amp_dig,
                        det_max,
                        amp_max,
                        w1_max,
                        learning_rate,
                        tau,
                        w1x):
        

        # Create a C double pointer for the best error output
        best_error_c = ctypes.c_double(0.0)

        # Call the Fortran function
        self.lib.run_grad_ascent(
            Np, band_dig, amp_dig, amp_max, det_max, w1_max, learning_rate, tau, 
            w1x, ctypes.byref(best_error_c)
        )
        
        # Convert C double to Python float
        best_error = 1 - best_error_c.value

        # Save pulse as instance of ShapedPulse class
        amps = w1x
        shaped_pulse = pulses.shaped_pulse(tau, amps, det_max, amp_max, best_error)
        
        return shaped_pulse


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
                                       f"libs/backend_lib{lib_ext}")
        
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
                      w1_max,
                      lambda_val,
                      tau):
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
            pulse_length, n_max, band_dig, amp_dig, amp_max, det_max, w1_max, lambda_val, tau, 
            best_sin_coeffs, best_cos_coeffs, ctypes.byref(best_error_c)
        )
        
        # Convert C double to Python float
        best_error = best_error_c.value

        # Save pulse as instance of ShapedPulse class
        times = np.linspace(0, tau, pulse_length)
        amps = self.coeffs_to_pulse(best_sin_coeffs, best_cos_coeffs, times, tau)
        shaped_pulse = pulses.shaped_pulse(tau, amps, det_max, amp_max, best_error)
        
        return shaped_pulse
    
    def coeffs_to_pulse(self, sin_coeffs, cos_coeffs, t, tau):  

        w = 2*np.pi/tau
        w1 = cos_coeffs[0]
    
        for i in range(len(sin_coeffs)):
            w1 += cos_coeffs[i+1]*np.cos((i+1)*w*t)
            w1 += sin_coeffs[i]*np.sin((i+1)*w*t)
    
        return w1 * w