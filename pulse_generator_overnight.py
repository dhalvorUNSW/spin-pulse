import numpy as np
import matplotlib.pyplot as plt
from optimisation.backend_wrapper import SimulatedAnnealing, GradientAscent
from qubits import *
from simulator import *
from pulses import *
import time

# test_pulse = mat_to_pulse('250kHz_7MHz/250ns_20250708_152350.mat')
save_folder = 'q2pulses/1MHz_30MHz/amp_corrected'

# Constants
dt = 1e-9
band_dig_1 = 11 #21
band_dig_2 = 31
amp_dig = 1
amp_max = 0
det_max = 1e6
w1_max = 2*np.pi*30e6
l = 1e3
learning_rate = 1e16
f_max = 300e6

trial_nums = 5
pulse_taus = 16e-9 * np.arange(1, 25, 1)
pulse_list = []
error_list = []

print('Starting pulse shaping runs.\n'
       'Constraints:\n'
       f'Peak amplitude: {round(w1_max/(1e6*2*np.pi), 3)} MHz\n'
       f'Band width = +/- {round(det_max/1e6 ,3)} MHz\n'
       '\n'
       f'Generating {len(pulse_taus)} times from {pulse_taus[0]/1e-9} ns to {pulse_taus[-1]/1e-9} ns.\n'
       f'Taking best of {trial_nums} trials at each time.')

for n in range(len(pulse_taus)):

    pulse_list = []
    error_list = []
    
    tau = pulse_taus[n]
    Np = int(np.ceil(tau/dt))
    n_max = min(30, int(np.floor(tau*f_max)))
    print('-------------------------------')
    print(f'Starting {tau/1e-9} ns trials.')

    for i in range(trial_nums):
        start = time.time()

        opt = SimulatedAnnealing()
        grad = GradientAscent()
        pulse = opt.run_annealing(Np, n_max, band_dig_1, amp_dig, amp_max, det_max, w1_max, l, tau)
        pulse = grad.run_grad_ascent(Np, band_dig_2, amp_dig, det_max, amp_max, w1_max, learning_rate, tau, w1x=pulse.amps)
        pulse_list.append(pulse)
        error_list.append(pulse.best_error)

        end = time.time()
        # Calculate the duration
        elapsed_time = end - start
        # Convert to hours, minutes, and seconds
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print(f'Run {i} done in {hours}h {minutes}min {seconds}s. Best error= {pulse.best_error}.')

    best_index  = np.argmin(error_list)
    best_pulse = pulse_list[best_index]
    best_pulse.save_to_mat(folder=save_folder)

    print(f'{round(tau/1e-9, 3)} ns trial done. Best error= {error_list[best_index]}')



