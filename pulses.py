import numpy as np
import matplotlib.pyplot as plt
import os
import json

class ShapedPulse:

    def __init__(self, amps, tau, pm_det, pm_amp, best_error):

        self.amps = amps # Array of amplitudes in pulse
        self.tau = tau # Length of pulse in seconds
        self.pulse_length = len(amps) # Number of points in pulse
        self.pm_det = pm_det # Width of frequency detuning pulse is robust to
        self.pm_amp = pm_amp # Width of amplitude change pulse is robust to
        self.amp_max = max(amps) # maximum amplitude of pulse
        self.best_error = best_error # Best error calculated by cost function

    # Encoding function for saving pulse
    def encode_pulse(self):
        return {
                'amps' : list(self.amps),
                'tau' : self.tau,
                'pulse_length' : self.pulse_length,
                'pm_det' : self.pm_det,
                'pm_amp' : self.pm_amp,
                'amp_max' : self.amp_max,
                'best_error' : self.best_error
            }
    
    def plot_pulse(self):
        dt = self.tau / self.pulse_length
        times = np.arange(dt, self.tau + self.dt, dt)

        plt.figure(figsize=(7, 4))
        plt.plot(times/1e-9, self.amps/(2*np.pi*1e6))
        plt.xlabel('Time (ns)')
        plt.ylabel('Pulse amp (MHz)')
        plt.title(r'Shaped pulse')
        plt.show()

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
