import numpy as np
import matplotlib.pyplot as plt

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

class FourierPulse:

    def __init__(self, tau, dt, time_symmetric=True):
        self.time_symmetric = time_symmetric # Only Cos coefficients
        self.tau = tau # Time length of pulse
        self.dt = dt # Sample steps in the pulse
        self.length = int(np.ceil(tau/dt))

        self.cos_coeffs = np.genfromtxt('/home/daniel/Projects/Random/FortranPulse/100_01tau_p5_coeffs.csv', delimiter=',')

        # Construct pulse from coeffs
        self.coeffs_to_pulse()

    # @property
    # def cos_coeffs(self):
    #     coeffs = np.genfromtxt('/home/daniel/Projects/Random/FortranPulse/Xpi2_1tau_15.csv', delimiter=',')
    #     return coeffs

    @property
    def full_coeffs(self):

        c_coeffs = np.array([-0.74170879870490003, 
                    0.44250378321054457, 
                    -1.9342070799542541,
                    0.69858867241981759, 
                    -3.4174180813195161, 
                    0.19340803782232863, 
                    0.75131121570836645, 
                    2.0912801297414858, 
                    0.42114672704852507, 
                    -0.46654074865506834, 
                    0.40969411054108834
        ])
        s_coeffs = np.array([9.2673617038559558e-2, 
                    -1.8732863462265226, 
                    1.7219381862779817, 
                    5.6693855700780248e-2, 
                    0.45588998095489575, 
                    1.1142471581716387, 
                    -1.2380100312434619, 
                    -0.38038273357031765, 
                    0.98800600116578241, 
                    0.60549825408730118
        ])

        return [c_coeffs, s_coeffs]

    def coeffs_to_pulse(self):

        # Construct pulse array from fourier coefficients.
        w = 2*np.pi/self.tau
        times = np.linspace(0, self.tau, self.length)
        w1 = np.ones(len(times))

        if self.time_symmetric == True:
            # Construct time symmetric pulse, coeffs = [0, 1, 2, ...]
            cos_coeffs = self.cos_coeffs
            n_max = len(cos_coeffs) - 1
            w1 = w1*cos_coeffs[0]

            for i in range(1, n_max + 1):
                w1 = w1 + cos_coeffs[i] * np.cos(i * w * times)

            w1 = w1 * w # Scale by 1/tau
        else:
            # Constuct general fourier pulse, coeffs = [cos_coeffs, sin_coeffs]
            [cos_coeffs, sin_coeffs] = self.full_coeffs
            n_max = len(cos_coeffs) - 1
            # Check
            if len(cos_coeffs) == (len(sin_coeffs) + 1):
                pass
            else:
                raise ValueError("Cos and sin coefficients missmatch.")
            
            w1 = w1*cos_coeffs[0]
            for t in range(len(times)):
                for i in range(1, n_max + 1):
                    w1[t] = w1[t] + cos_coeffs[i] * np.cos(i * w * times[t])
                    w1[t] = w1[t] + sin_coeffs[i - 1] * np.sin(i * w * times[t])

            w1 = w1 * w # Scale by 1/tau

        # Set pulse amps
        self.amps = w1

    def plot_pulse(self):
        times = np.arange(self.dt, self.tau + self.dt, self.dt)

        fig = plt.figure(figsize=(7, 4))
        plt.plot(times/1e-9, self.amps/(2*np.pi*1e6))
        plt.xlabel('Time (ns)')
        plt.ylabel('Pulse amp (MHz)')
        plt.title(r'Shaped $X_{\pi/2}$ pulse')
        plt.show()
