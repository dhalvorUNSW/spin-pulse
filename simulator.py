import numpy as np
import qutip as q
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from optimisation.backend_wrapper import evolveState_fast

from qubits import SpinQubit
from pulses import cw_pulse, PulseSequence

C1 = '#34A300' # Green
C2 = '#DD404B' # Red
C3 = '#628395' # Blue
C4 = '#B57E2C' # Brown
C5 = '#360568' # Purple
colors = [C3, C4, C5]

def amp_rabi_map(pulse_sequence, amplitudes, detunings, plot_output=True):
    """
    Generate map of detuning vs pulse amplitude for arb pulse sequence.
    """

    projections = np.zeros([len(amplitudes), len(detunings)])
    original_amps = pulse_sequence.amps

    for d in tqdm(range(len(detunings))):
        for a in range(len(amplitudes)):
            q1 = SpinQubit()
            pulse_sequence.det = detunings[d]
            pulse_sequence.amps = original_amps * amplitudes[a]
            states = evolveState(q1, pulse_sequence)
            t_projs = plotProjection(states, pulse_sequence, plot_output=False)
            projections[a, d] = t_projs[-1]

    projections = np.flip(projections, 0)

    if plot_output == True:
        fig, ax = plt.subplots()
        im = ax.imshow(projections, interpolation=None,
                       extent=[detunings.min()/1e6, detunings.max()/1e6, 0, amplitudes[-1]])
        ax.set_aspect('auto')
        ax.set_xlabel(r'Detuning: $f - f_0 (MHz)$')
        ax.set_ylabel('Amps')
        fig.colorbar(im, ax=ax, label=r"$|0\rangle$ probability")

    return projections

def cw_rabi_chevron(fRabi, times, detunings, filtering=False, plot_output=True):
    """
    Generate rabi chevron for cw pulse (fixed amplitude).
    """

    wRabi = 2*np.pi*fRabi
    projections = np.zeros([len(times), len(detunings)])

    for d in tqdm(range(len(detunings))):
        det = detunings[d]

        if filtering == False:
            # Generate pulse
            dt = times[1] - times[0]
            pulse = cw_pulse(times[-1], wRabi, sample_rate=1/dt)
            sequence = PulseSequence(det=det, dt = dt)
            sequence.add_pulse(pulse)
            # Simulate pulse
            q = SpinQubit()
            states = evolveState(q, sequence)
            projections[:, d] = plotProjection(states, sequence, plot_output=False)

        else:
            # Generate filtered pulse
            for t in range(len(times)):
               # Generate pulse
                dt_in = times[1] - times[0]
                pulse = cw_pulse(times[t], wRabi, sample_rate=1/dt_in)
                sequence = PulseSequence(det=det, dt = dt_in)
                sequence.add_pulse(pulse)
                sequence.add_pause(dt_in) # Add sufficient pause for ring-down
                sequence.gaussian_filter(output_dt=1e-10, fc=500e6)

                q = SpinQubit()
                states = evolveState(q, sequence)
                t_projs = plotProjection(states, sequence, plot_output=False)
                projections[t, d] = t_projs[-1]

    
    projections = np.flip(projections, 0)
    if plot_output == True:
        fig, ax = plt.subplots()
        im = ax.imshow(projections, interpolation=None,
                       extent=[detunings.min()/1e6, detunings.max()/1e6, 0, times[-1]/1e-9])
        ax.set_aspect('auto')
        ax.set_xlabel(r'Detuning: $f - f_0 (MHz)$')
        ax.set_ylabel('Time (ns)')
        fig.colorbar(im, ax=ax, label=r"$|0\rangle$ probability")

    return projections

def projection_spectrum(pulse_sequence, detunings, plot_output=True, log_scale=False):

    projections = np.zeros(len(detunings))
    
    for d in range(len(detunings)):
        q = SpinQubit()
        pulse_sequence.det = detunings[d]
        states = evolveState(q, pulse_sequence)
        t_projs = plotProjection(states, pulse_sequence, plot_output=False)
        projections[d] = t_projs[-1]

    if plot_output:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(detunings/1e6, projections, color=C5)
        plt.xlabel('Detuning (MHz)')
        plt.ylabel(r'State projection: $|\langle \phi | \psi (t) \rangle|^2$')
        plt.title('State projection after pulse.')
        if log_scale:
            plt.yscale('log')

    return projections

def evolveState(qubit, pulse_sequence):
    """
    Evoles state of qubit given qubit unitary through pulse sequence.
    """
    evolve_wrapper = evolveState_fast()
    Np = len(pulse_sequence.amps)
    w1x = pulse_sequence.amps
    w1y = np.zeros(Np)
    dt = pulse_sequence.dt
    det = np.ones(Np) * pulse_sequence.det

    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j])  # Initial state
    states = evolve_wrapper.evolve(dt, w1x, w1y, det, psi0)
    
    # N = len(pulse_sequence.amps)
    # # Temporary single axis
    # w1x = pulse_sequence.amps
    # w1y = np.zeros(N)
    # dt = pulse_sequence.dt
    # det = pulse_sequence.det

    # states = np.zeros((len(w1x), 2, 1), dtype=complex)

    # for i in range(N):

    #     U = qubit.unitary(w1x[i], w1y[i], det, dt)
    #     qubit.state = np.matmul(U, qubit.state)
    #     states[i, :, :] = qubit.state

    return states

def plotBlochSphere(states):
        
    # Plot states on Bloch sphere
    points = np.array([state_to_point(s) for s in states]).T
    gradient = [colorGradient(C1,C2, x/len(states)) for x in range(len(states))]
    b = q.Bloch()
    b.add_points(points)
    b.point_color = gradient
    b.frame_alpha = 0.1
    
    return b


def plotProjection(states, pulse_sequence, proj="up", plot_output=True):
    # Set state to project onto
    if proj == "up":
        proj_state = np.array([[1], [0]], dtype=complex)
    elif proj == "down":
        proj_state = np.array([[0], [1]], dtype=complex)
    else:
        raise ValueError("Projection state unrecognised: choose 'up' or 'down'.")
    
    # Project states onto desired state 'proj'
    if states.shape[1] == 1:
        P  = np.abs(np.matmul(np.conjugate(proj_state).T, states))**2
    else:
        times = np.linspace(0, pulse_sequence.tau, len(states))
        P = np.zeros(len(states))
        for s in range(len(states)):
            P[s] = np.abs(np.matmul(np.conjugate(proj_state).T, states[s]))**2

    if plot_output:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(times, P, color=C5)
        plt.xlabel('time (s)')
        plt.ylabel(r'State projection: $|\langle \phi | \psi (t) \rangle|^2$')
        plt.title('State projection during pulse.')
        plt.ylim([-0.1, 1.1])

    return P


def state_to_point(vector):
    a = complex(vector[0])
    b = complex(vector[1])
    
    phi = np.angle(b) - np.angle(a)
    theta = np.arccos(np.abs(a))*2 # Polar angle

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    return [x, y, z]


def colorGradient(c1,c2,r=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-r)*c1 + r*c2)


## IN DEVELOPMENT

def digitalRabiError(pulse, N_shots, det_noise_width, amp_noise_width):
    # Currently just one point
    # Generate noise
    det_noise_array = np.random.normal(loc=0, scale=det_noise_width, size=[N_shots])
    amp_noise_array = np.random.normal(loc=0, scale=amp_noise_width, size=[N_shots])

    # Simulate digital rabi for n_shots with noise
    proj = 0
    for n in tqdm(range(N_shots)):
        # Initialise qubit
        q1 = qubits.SpinQubit()

        rabi_sequence = pulses.PulseSequence(det_offset=0e6)
        # For now, try optimise projection after 8th pulse
        for _ in range(8):
            rabi_sequence.X2_custom(pulse)
        
        rabi_sequence.det_offset = det_noise_array[n]
        rabi_sequence.amp_offset = amp_noise_array[n]

        # Simulate pulse and get projection
        states = evolveState(q1, rabi_sequence)
        proj += plotProjection(states[-1], rabi_sequence, plot_output=False)[0][0]

    proj_avg = proj/N_shots

    # Evaluate cost function
    E = -proj_avg

    return E