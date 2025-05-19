#%%
from qubits import SpinQubit
import pulses
from simulator_test import evolveState, plotBlochSphere, plotProjection
from noise import BroadbandNoise
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Initialise qubit
q1 = SpinQubit(initial_state="up")

# Define pulse sequence
fRabi = 2.5e6 # Driving speed of continuous pulse
tau_X2 = 0.25*1/fRabi 
tau = 1 * 1/fRabi # rabi oscillations
dt = 0.1e-9 # Step size of simulation

sequence = pulses.PulseSequence(det_offset=0, dt=dt)
sequence.cw(tau, fRabi)
# sequence.wait(5*tau_X2)
# sequence.cw(tau_X2, fRabi)

# Define noise model 
# Noise model should be an object with settable parameters that the sim function can call to add noise
noise_model = BroadbandNoise(sequence, exponent=2)
# noise_model.plot_time_series()
# noise_model.plot_spectrum()
sequence.detuning_noise_width = 0e6 # 1MHz noise on detuning
# sequence.amp_noise_width = 0.5e6/4e9 * sequence.detuning_noise_width
sequence.amp_noise_width = 5e6

q1 = SpinQubit()
states = evolveState(q1, sequence, noise_model)
plotBlochSphere(states)

#%%
N_shots = 50
proj_array = np.zeros([N_shots, sequence.length])
for n in tqdm(range(N_shots)):
    # Reinitialise qubit each time
    q1 = SpinQubit(initial_state="up")
    # Simulate qubit evolution given qubit, pulse sequence and noise model
    states = evolveState(q1, sequence, noise_model)
    proj_array[n, :] = plotProjection(states, sequence, proj="up", plot_output=False)

proj_avg = np.average(proj_array, 0)
times = np.linspace(sequence.dt, sequence.time  + sequence.dt, sequence.length)


# fig = plt.figure(figsize=(7, 4))
# for n in range(N_shots):
#     plt.plot(times, proj_array[n, :])

plt.plot(times, proj_avg)
plt.xlabel('time (s)')
plt.ylabel(r'State projection: $|\langle \phi | \psi (t) \rangle|^2$')
plt.title('State projection during pulse.')
plt.ylim([-0.1, 1.1])
plt.show()
