import numpy as np
import matplotlib.pyplot as plt
from optimisation.optimisers import SimulatedAnnealing

# Set global parameters kept constant
n_max = 5
tau = 50e-9
pulse_length = 50
band_dig = 6
amp_dig = 1
amp_max = 1
det_max = 0.5/tau
init_temp = 100
w1_max = 2*np.pi*40e6
lambda_val = 1000

# Test
cooling_rate = 0.
opt = SimulatedAnnealing()
pulse =  opt.run_annealing(
    n_max=n_max,
    pulse_length=pulse_length,
    band_dig=band_dig,
    amp_dig=amp_dig,
    amp_max=amp_max,
    det_max=det_max,
    init_temp=init_temp,
    cooling_rate=cooling_rate,
    w1_max=w1_max,
    lambda_val=lambda_val,
    tau=tau,
    save_pulse=False
)

