import numpy as np
import matplotlib.pyplot as plt
from optimisation_algorithims.annealing_optimiser import SimulatedAnnealing

# Set global parameters kept constant
n_max = 15
tau = 100e-9
pulse_length = 100
band_dig = 8
amp_dig = 1
amp_max = 1
det_max = 0.5/tau
init_temp = 10
w1_max = 2*np.pi*40e6

# Define parameters to be changed
cooling_rate_array = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
results_array = np.zeros([3, len(cooling_rate_array)])

for c in range(len(cooling_rate_array)):
    for i in range(3):
        cooling_rate = cooling_rate_array[c]
        opt = SimulatedAnnealing()
        coeffs, error =  opt.run_annealing(
            n_max=n_max,
            pulse_length=pulse_length,
            tau=tau,
            band_dig=band_dig,
            amp_dig=amp_dig,
            amp_max=amp_max,
            det_max=det_max,
            init_temp=init_temp,
            cooling_rate=cooling_rate,
            w1_max=w1_max
        )

        