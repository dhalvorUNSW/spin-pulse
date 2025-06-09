import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve

class Pulse:
    def __init__(self, tau, sample_rate=1e9):
        self.tau = tau # Length of pulse in time
        self.sample_rate = sample_rate # Sample rate of pulse (default of 1GS/s)
        # Generate extra properties
        self.dt = 1/sample_rate # Time steps of pulse

    def generate_pulse_amps(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def plot_pulse(self):
        if self.amps is None:
            raise ValueError("Amplitudes not generated yet.")
                
        times = np.arange(self.dt, self.tau + self.dt, self.dt)
        plt.figure(figsize=(7, 4))
        plt.plot(times/1e-9, self.amps/(2*np.pi*1e6))
        plt.xlabel('Time (ns)')
        plt.ylabel(r'Pulse amplitude, $\omega/2\pi$ (MHz)')
        plt.grid(True)
        plt.show()

class cw_pulse(Pulse):
    def __init__(self, tau, peak_amp, sample_rate=1e9):
        super().__init__(tau, sample_rate)
        self.peak_amp = peak_amp # Peak amplitude of pulse in rad/s
        self.generate_pulse_amps()

    def generate_pulse_amps(self):
        self.amps = self.peak_amp * np.ones(int(np.ceil(self.tau/self.dt)))

        

class PulseSequence:

    def __init__(self, dt=1e-9):
        self.amps = np.array([])
        self.tau = 0
        self.dt = dt

    def update_times(self):
        self.times = np.arange(self.dt, self.tau + self.dt, self.dt)

    def add_pulse(self, pulse):

        if self.dt is None:
            self.dt = pulse.dt
        elif pulse.dt != self.dt:
            raise ValueError("Added pulse time-step not consistent with pulse sequence dt. \n" \
            f"Please redefine pulse sequence with correct dt = {pulse.dt}")
        
        self.amps = np.concatenate([self.amps, pulse.amps])
        self.tau += pulse.tau 
        self.update_times()

    def add_pause(self, pause_time):
        """
        Adds wait time at defined sample rate.

        Args:
            length: Length of pulse in s.
        """
        if self.dt is None:
            self.dt = 1e-9

        pause = np.zeros(int(np.ceil(pause_time/self.dt)))
        self.amps = np.concatenate([self.amps, pause])
        self.tau += pause_time
        self.update_times()

    def gaussian_filter(self, output_dt=0.1e-9, fc=500e6, turn_off=False):
        """
        Apply a *causal* Gaussian low-pass filter approximating a fc bandwidth
        to an input waveform using convolution.
        """

        # Define time arrays
        t_input = np.arange(len(self.amps)) * self.dt
        t_out = np.arange(0, t_input[-1] + self.dt, output_dt)

        # Interpolate waveform using 'previous' to keep step-like transitions
        interpolator = interp1d(t_input, self.amps, kind='previous', fill_value="extrapolate")
        amps_upsampled = interpolator(t_out)
        # Add zero to first point in waveform
        amps_upsampled = np.insert(amps_upsampled, 0, 0.0)

        # Gaussian parameters
        sigma_t = 1 / (2 * np.pi * fc)

        # Define causal Gaussian: t >= 0 only
        t_kernel = np.arange(0, 5 * sigma_t + output_dt, output_dt)
        gaussian_kernel = np.exp(-0.5 * (t_kernel / sigma_t)**2)

        # Normalize to preserve amplitude
        gaussian_kernel /= np.sum(gaussian_kernel)

        # Convolve and trim to match output length
        y_full = convolve(amps_upsampled, gaussian_kernel, mode='full')

        # Update sequence parameters
        self.amps = y_full[:len(t_out)] # Remove zero point
        self.times = t_out[:] # Remove zero point
        self.dt = output_dt

    def plot_pulse(self):

        if self.amps is None:
            raise ValueError("Amplitudes not generated yet.")
        
        plt.figure(figsize=(7, 4))
        plt.plot(self.times/1e-9, self.amps/(2*np.pi*1e6))
        plt.xlabel('Time (ns)')
        plt.ylabel(r'Pulse amplitude, $\omega/2\pi$ (MHz)')
        plt.grid(True)
        plt.show()



