import numpy as np
import matplotlib.pyplot as plt
import pulses

class BroadbandNoise:

    def __init__(self, length, exponent=0):

        self.exponent = exponent # Controls PSD slope
        self.length = length

        # TODO: implement "generate_noise" function for all noise types in model
        self.generate_broadband_noise()

    def generate_broadband_noise(self):
        """
        Generate noise with a specific frequency spectrum.
        (written by J. Huang)

        Args:
            - num_samples: Number of points in the signal.
            - exponent: Controls the power spectral density (PSD) slope.
            - 0  -> White noise (flat spectrum)
            - 1 -> Pink noise (1/f spectrum)
            - 2 -> Brown noise (1/f^2 spectrum)
        """
        N = self.length

        # Generate random Fourier coefficients
        freqs = np.fft.rfftfreq(N)  # Positive frequency components
        amplitudes = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))  # Complex noise

        # Apply frequency-dependent scaling
        with np.errstate(divide='ignore', invalid='ignore'):
            scaling = np.where(freqs == 0, 0, 1 / (freqs ** (self.exponent / 2.0)))  # PSD ~ 1/f^exponent
            amplitudes *= scaling

        # Convert back to time domain
        self.amps = np.fft.irfft(amplitudes, N)
        # Normalize to unit variance
        self.amps /= np.std(self.amps)

    def plot_time_series(self):
        """
        Plot time series of noise generated.
        """
        times = np.arange(self.dt, self.time + self.dt, self.dt)

        fig = plt.figure(figsize=(7, 4))
        plt.plot(times, self.amps)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (a.u.)")
        plt.title(f"Broadband Noise, exp={self.exponent}")
        plt.grid()
        plt.show()

    def plot_spectrum(self):
        """
        Plot power spectrum of noise.
        """
        N = int(np.ceil(self.length/self.dt))
        self.psd = np.abs(np.fft.rfft(self.amps)) ** 2
        freqs = np.fft.rfftfreq(N)

        fig = plt.figure(figsize=(7, 4))
        plt.loglog(freqs[1:], self.psd[1:])
        plt.xlabel("Frequency")
        plt.ylabel("Power spectral density")
        plt.title(f"PSD of Broadband noise, exp={self.exponent}")
        plt.grid()
        plt.show()

