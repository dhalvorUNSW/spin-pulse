from optimisation.backend_wrapper import GradientAscent
import numpy as np
from qubits import *
from simulator import *
from pulses import *

test_pulse = mat_to_pulse('500kHz_13MHz/200ns_20250619_125305.mat')
w1 = 2*np.pi *1/(800e-9)
# test_pulse = cw_pulse(200e-9, w1)
Np = len(test_pulse.amps)
band_dig = 31
amp_dig = 1
amp_max = 0
tau = 200e-9
# det_max = 0.1/tau
det_max = 0.5e6
learning_rate = 2e16

ga = GradientAscent()
ga.run_grad_ascent(Np, band_dig, amp_dig, det_max, amp_max, learning_rate, tau, w1x_init=test_pulse.amps)
