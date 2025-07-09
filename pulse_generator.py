import numpy as np
import matplotlib.pyplot as plt
from optimisation.backend_wrapper import SimulatedAnnealing, GradientAscent
from qubits import *
from simulator import *
from pulses import *
import time

# test_pulse = mat_to_pulse('250kHz_7MHz/250ns_20250708_152350.mat')
save_folder = 'q2pulses/100kHz_30MHz/amp_corrected'

# Constants
dt = 1e-9
band_dig_1 = 5 #21
band_dig_2 = 31
amp_dig = 5
amp_max = 0.01 # 1 % pulse amp
det_max = 0.1e6
w1_max = 2*np.pi*30e6
l = 1e3
learning_rate = 1e16
f_max = 300e6

trial_nums = 5
pulse_taus = 16e-9 * np.arange(1, 25, 1)
    
tau = 100e-9
Np = int(np.ceil(tau/dt))
n_max = min(30, int(np.floor(tau*f_max)))
print('-------------------------------')
print(f'Starting {tau/1e-9} ns trials.')

# start = time.time()

opt = SimulatedAnnealing()
grad = GradientAscent()
pulse = opt.run_annealing(Np, n_max, band_dig_1, amp_dig, amp_max, det_max, w1_max, l, tau)
# pulse = grad.run_grad_ascent(Np, band_dig_2, amp_dig, det_max, amp_max, w1_max, learning_rate, tau, w1x=pulse.amps)

# end = time.time()
# Calculate the duration
# elapsed_time = end - start
# Convert to hours, minutes, and seconds
# hours = int(elapsed_time // 3600)
# minutes = int((elapsed_time % 3600) // 60)
# seconds = int(elapsed_time % 60)
# print(f'Run done in {hours}h {minutes}min {seconds}s. Best error= {pulse.best_error}.')
print(f'Best error = {pulse.best_error}')

pulse.save_to_mat(folder=save_folder)