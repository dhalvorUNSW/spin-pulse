import numpy as np
import matplotlib.pyplot as plt

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
        plt.plot(times, self.amps/(2*np.pi))
        plt.xlabel('Time (s)')
        plt.ylabel(r'Pulse amplitude, $\omega/2\pi$ (Hz)')
        plt.grid(True)
        plt.show()

class cwPulse(Pulse):
    def __init__(self, tau, peak_amp, sample_rate=1e9):
        super().__init__(tau, sample_rate)
        self.peak_amp = peak_amp # Peak amplitude of pulse in rad/s
        self.generate_pulse_amps()

    def generate_pulse_amps(self):
        self.amps = self.peak_amp * np.ones(int(np.ceil(self.tau/self.dt)))

    
        

class PulseSequence:

    def __init__(self, amp_offset=0.0, det_offset=0.0, dt=1e-9):
        self.amps = np.array([])
        self.amp_noise = np.array([])
        self.dt = dt # Step size in pulse sequence
        self.det_offset = det_offset # Detuning of pulse
        self.amp_offset = amp_offset

    @property
    def length(self):
        """
        Property that returns the length of the pulse sequence.
        """
        return len(self.amps)
    
    @property
    def time(self):
        """
        Property that returns the length in time of the pulse sequence.
        """
        return self.length * self.dt

    def play(self, pulse):
        """
        Add a given pulse to the sequence
        """
        self.amps = np.concatenate([self.amps, pulse])
        self.det = self.det_offset*np.ones(len(self.amps)) # Resize det array

    ## Library of pulse shapes
    def wait(self, tau):
        """
        Adds wait time at defined sample rate.

        Args:
            length: Length of pulse in s.
        """
        pulse = np.zeros(int(np.ceil(tau/self.dt)))
        self.play(pulse)

    def cw(self, tau, drive_speed):
        """
        Adds constant pulse at defined sample rate and amplitude.

        Args:
            length: Length of pulse in s.
            drive_speed: Drive speed pulse in units of Hz (rabi frequency).
        """
        amplitude = 2 * np.pi * drive_speed
        pulse = amplitude * np.ones(int(np.ceil(tau/self.dt)))
        self.play(pulse)

    def X2_custom(self, shaped_pulse): 
        # TODO: add some check that shaped_pulse is compatable with 
        # self.play(shaped_pulse.amps)
        pulse = shaped_pulse.amps
        self.play(pulse)
