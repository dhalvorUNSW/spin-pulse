import numpy as np
import qutip as q
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

C1 = '#34A300' # Green
C2 = '#DD404B' # Red
C3 = '#628395' # Blue
C4 = '#B57E2C' # Brown
C5 = '#360568' # Purple
colors = [C3, C4, C5]

def evolveState(qubit, pulse_sequence, det_noise):

    if len(pulse_sequence.amps) != len(det_noise):
        raise ValueError('Pulse amps and det noise not same length.')

    # Temporary single axis
    w1x = pulse_sequence.amps + pulse_sequence.amp_offset
    w1y = np.zeros(pulse_sequence.length)
    det = pulse_sequence.det_offset*np.ones(len(w1x)) + det_noise
    dt = pulse_sequence.dt

    # if interpolate == True:
    #     sr = 50e9
    #     NUM_STEPS = int(sr*pulse_sequence.time)
    #     times_interp = np.linspace(1/sr, pulse_sequence.time, NUM_STEPS)
    #     times = np.linspace(pulse_sequence.dt, pulse_sequence.time, pulse_sequence.length)
    #     w1x = np.interp(times_interp, times, w1x)
    #     w1y = np.interp(times_interp, times, w1y)

    states = np.zeros((len(w1x), 2, 1), dtype=complex)

    for i in range(pulse_sequence.length):
        U = qubit.unitary(w1x[i], w1y[i], det[i], dt)
        qubit.state = np.matmul(U, qubit.state)
        states[i, :, :] = qubit.state

    return states

def plotBlochSphere(states):
        
    # Plot states on Bloch sphere
    points = np.array([state_to_point(s) for s in states]).T
    gradient = [colorGradient(C1,C2, x/len(states)) for x in range(len(states))]
    b = q.Bloch()
    b.add_points(points)
    b.point_color = gradient
    b.frame_alpha = 0.1
    b.show()
    mpl.pyplot.show()
    
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
        times = np.linspace(0, pulse_sequence.time, len(states))
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