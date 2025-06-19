import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as q
import scipy.constants as c
from scipy.linalg import expm
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

### Set constants used below

hbar = c.hbar
muB = c.value('Bohr magneton')

sigmax = np.array([[0, 1], 
                   [1, 0]])
sigmay = np.array([[0, -1.0j], 
                   [1.0j, 0]])
sigmaz = np.array([[1, 0], 
                   [0, -1]])

C1 = '#34A300' # Green
C2 = '#DD404B' # Red
C3 = '#628395' # Blue
C4 = '#B57E2C' # Brown
C5 = '#360568' # Purple
colors = [C3, C4, C5]


### Plotting functions
# All plotting functions should have capability to plot as well as just return
# the data that is plotted.

def powerRabiSpectrum(
    psi0,
    proj,
    w1x,
    w1y,
    det,
    tau,
    amps,
    plot_output=True
):
    """
    TODO: broken
    """

    p_rabi_spectrum = np.zeros((len(amps), len(det)))

    for d in tqdm(range(len(det))):
        for p in range(len(amps)):
            x_pulse = w1x*amps[p]
            y_pulse = w1y*amps[p]
            states = evolveState(psi0, x_pulse, y_pulse, det[d], tau)
            p_rabi_spectrum[p, d] = np.abs(np.matmul(np.conjugate(proj).T, states[-1, :, :]))**2

    # Reshape
    p_rabi_spectrum = np.rot90(p_rabi_spectrum, 2)

    if plot_output == True:
        fig, ax = plt.subplots()
        im = ax.imshow(p_rabi_spectrum, interpolation=None,
                       extent=[det.min(), det.max(), 0, amps.max()], cmap=parula_map)
        ax.set_aspect('auto')
        ax.set_xlabel(r'Detuning: $f - f_0 (Hz)$')
        ax.set_ylabel('Amp ($\%$ of ideal)')
        fig.colorbar(im, ax=ax, label=r"$|0\rangle$ probability")

    return p_rabi_spectrum

def timeRabiSpectrum(
    psi0,
    proj,
    w1x,
    w1y,
    det,
    tau,
    plot_output=True
):
    """
    Generates 2D plot of initial state projection vs detuning and evoltion time.
    For now only supports square waves
    - tau: if list, times to calculate state at. If float or int, 
    """

    sr = 2/1e-9
    w1x = w1x*np.ones(int(sr*tau))
    w1y = w1y*np.ones(int(sr*tau))

    t_rabi_spectrum = np.zeros((len(w1x), len(det)))
    for d in range(len(det)):
        states = evolveState(psi0, w1x, w1y, det[d], tau)
        for s in range(len(states)):
            t_rabi_spectrum[s,d] = np.abs(np.matmul(np.conjugate(proj).T, states[s]))**2

    # Reshape
    t_rabi_spectrum = np.rot90(t_rabi_spectrum, 2)

    if plot_output == True:
        fig, ax = plt.subplots()
        im = ax.imshow(t_rabi_spectrum, interpolation=None,
                       extent=[det.min(), det.max(), 0, tau/1e-9], cmap=parula_map)
        ax.set_aspect('auto')
        ax.set_xlabel(r'Detuning: $f - f_0 (Hz)$')
        ax.set_ylabel('Time (ns)')
        # fig.colorbar(im, ax=ax, label=r"$|0\rangle$ probability")

    return t_rabi_spectrum

def projectionSpectrum(
    psi0, 
    proj,
    w1x,
    w1y,
    det,
    tau,
    plot_output=True
):
    
    if isinstance(w1x, (int, float)) and isinstance(w1y, (int, float)):
        # Square pulse input
        NUM_STEPS = 1000
        w1x = w1x*np.ones(NUM_STEPS)
        w1y = w1y*np.ones(NUM_STEPS)
    elif isinstance(w1x, (int, float)) and isinstance(w1y, (list, np.ndarray)):
        NUM_STEPS = len(w1y)
        w1x = w1x*np.ones(NUM_STEPS)
    elif isinstance(w1y, (int, float)) and isinstance(w1x, (list, np.ndarray)):
        NUM_STEPS = len(w1x)
        w1y = w1y*np.ones(NUM_STEPS)

    spectrum = np.zeros(len(det))
    for d in range(len(det)):
        final_state = evolveState(psi0, w1x, w1y, det[d], tau)[-1]
        spectrum[d] = np.abs(np.matmul(np.conjugate(proj).T, final_state))**2

    if plot_output == True:
        # Plot spectrum
        fig = plt.figure(figsize=(7, 4))
        plt.plot(det, spectrum, color=C4)
        plt.xlabel('Detuning (Hz)')
        plt.ylabel(r'State projection: $|\langle \phi | \psi (t) \rangle|^2$')
        plt.title('Final Projection vs. Detuning')
        # plt.ylim([-0.1, 1.1])

    return spectrum

def polarisationSpectrum(
    psi0,
    w1x,
    w1y,
    det,
    tau,
    polarisations=['Px', 'Py', 'Pz'],
    plot_output=True
):
    
    if isinstance(w1x, (int, float)) and isinstance(w1y, (int, float)):
        # Square pulse input
        NUM_STEPS = 1000
        w1x = w1x*np.ones(NUM_STEPS)
        w1y = w1y*np.ones(NUM_STEPS)
    elif isinstance(w1x, (int, float)) and isinstance(w1y, (list, np.ndarray)):
        NUM_STEPS = len(w1y)
        w1x = w1x*np.ones(NUM_STEPS)
    elif isinstance(w1y, (int, float)) and isinstance(w1x, (list, np.ndarray)):
        NUM_STEPS = len(w1x)
        w1y = w1y*np.ones(NUM_STEPS)

    # Calculate final polarisations at each detuning
    spectrum = np.zeros((len(polarisations), len(det)))

    for d in range(len(det)):
        final_state = evolveState(psi0, w1x, w1y, det[d], tau)[-1]
        for p in range(len(polarisations)):
            spectrum[p, d] = calcPolarisations(final_state, polarisations[p])
    
    if plot_output == True:
        # Plot spectrum
        fig = plt.figure(figsize=(7, 4))
        for p in range(len(polarisations)):
            plt.plot(det/1e6, spectrum[p, :], color=colors[p], label=polarisations[p])

        plt.xlabel('Detuning (MHz)')
        plt.ylabel(r'Polarisation')
        plt.legend()
        plt.title('Final Polarisation vs. Detuning')
        # plt.ylim([-1.1, 1.1])

    return spectrum

def simulatePolarisations( 
    psi0,
    proj,
    w1x,
    w1y,
    det,
    tau,
    polarisations=['Px','Py','Pz'],
    plot_output=True
):
    if isinstance(w1x, (int, float)) and isinstance(w1y, (int, float)):
        # Square pulse input
        NUM_STEPS = 1000
        w1x = w1x*np.ones(NUM_STEPS)
        w1y = w1y*np.ones(NUM_STEPS)
    elif isinstance(w1x, (int, float)) and isinstance(w1y, (list, np.ndarray)):
        NUM_STEPS = len(w1y)
        w1x = w1x*np.ones(NUM_STEPS)
    elif isinstance(w1y, (int, float)) and isinstance(w1x, (list, np.ndarray)):
        NUM_STEPS = len(w1x)
        w1y = w1y*np.ones(NUM_STEPS)

    states = evolveState(psi0, w1x, w1y, det, tau)

    # Calculate polarisations of states
    times = np.linspace(0, tau, len(states))
    Pol = np.zeros((len(polarisations), len(states)))

    for p in range(len(polarisations)):
        Pol[p, :] = calcPolarisations(states, polarisations[p])
    
    if plot_output == True:
        # Plot polarisations
        fig = plt.figure(figsize=(7, 4))
        for p in range(len(polarisations)):
            plt.plot(times, Pol[p, :], color=colors[p], label=polarisations[p])

        plt.xlabel('time (s)')
        plt.ylabel(r'Polarisation')
        plt.legend()
        plt.title('State polarisation during pulse.')
        plt.ylim([-1.1, 1.1])

    return Pol



def simulateProjection(
    psi0,
    proj,
    w1x,
    w1y,
    det,
    tau,
    plot_output=True
):
    if isinstance(w1x, (int, float)) and isinstance(w1y, (int, float)):
        # Square pulse input
        NUM_STEPS = 1000
        w1x = w1x*np.ones(NUM_STEPS)
        w1y = w1y*np.ones(NUM_STEPS)
    elif isinstance(w1x, (int, float)) and isinstance(w1y, (list, np.ndarray)):
        NUM_STEPS = len(w1y)
        w1x = w1x*np.ones(NUM_STEPS)
    elif isinstance(w1y, (int, float)) and isinstance(w1x, (list, np.ndarray)):
        NUM_STEPS = len(w1x)
        w1y = w1y*np.ones(NUM_STEPS)

    states = evolveState(psi0, w1x, w1y, det, tau)

    # Project states onto desired state 'proj'
    times = np.linspace(0, tau, len(states))
    P = np.zeros(len(states))
    for s in range(len(states)):
        P[s] = np.abs(np.matmul(np.conjugate(proj).T, states[s]))**2

    if plot_output == True:
        fig = plt.figure(figsize=(7, 4))
        plt.plot(times, P, color=C5)
        plt.xlabel('time (s)')
        plt.ylabel(r'State projection: $|\langle \phi | \psi (t) \rangle|^2$')
        plt.title('State projection during pulse.')
        plt.ylim([-0.1, 1.1])

    return P



def simulateBlochSphere(
    psi0,
    w1x,
    w1y, 
    det, 
    tau, 
    plot_output=True,
    interpolate=True
):
    if isinstance(w1x, (int, float)) and isinstance(w1y, (int, float)):
        # Square pulse input
        NUM_STEPS = 1000
        w1x = w1x*np.ones(NUM_STEPS)
        w1y = w1y*np.ones(NUM_STEPS)
    elif isinstance(w1x, (int, float)) and isinstance(w1y, (list, np.ndarray)):
        NUM_STEPS = len(w1y)
        w1x = w1x*np.ones(NUM_STEPS)
    elif isinstance(w1y, (int, float)) and isinstance(w1x, (list, np.ndarray)):
        NUM_STEPS = len(w1x)
        w1y = w1y*np.ones(NUM_STEPS)

    if interpolate == True:
        sr = 50e9
        NUM_STEPS = int(sr*tau)
        times_interp = np.linspace(1/sr, tau, NUM_STEPS)
        times = np.linspace(1e-9, tau, len(w1x))
        w1x = np.interp(times_interp, times, w1x)
        w1y = np.interp(times_interp, times, w1y)

    states = evolveState(psi0, w1x, w1y, det, tau)
    
    # Plot states on Bloch sphere
    points = np.array([state_to_point(s) for s in states]).T
    gradient = [colorGradient(C1,C2, x/len(states)) for x in range(len(states))]
    
    b = q.Bloch()
    b.add_points(points)
    b.point_color = gradient
    b.frame_alpha = 0.1
    if plot_output == True:
        b.show()
    
    return b


def evolveState(psi0, w1x, w1y, det, tau):

    states = np.zeros((len(w1x), 2, 1), dtype=complex)
    dt = tau/len(w1x)

    psi = psi0
    for i in range(len(w1x)):
        U = hardPulsePropagator(w1x[i], w1y[i], det, dt)
        psi = np.matmul(U, psi)
        states[i, :, :] = psi

    return states

def hardPulsePropagator(w1x, w1y, det, dt):
    det = 2*np.pi*det # Convert to units of radians
    weff = np.sqrt(w1x**2 + w1y**2 + det**2)
    beta = weff*dt # Rotation angle
    if weff == 0:
        U = np.eye(2, dtype=complex)
    else:
        U = np.cos(beta/2)*np.eye(2, dtype=complex) - 1.0j*np.sin(beta/2)*(w1x/weff*sigmax + w1y/weff*sigmay + det/weff*sigmaz)

    return U



# def hardPulsePropagator(b1, g, dt, det):

#     gamma = g*muB/hbar
#     # Rotating frame hamiltonian:
#     H = np.array([[2*np.pi*det, gamma*b1/2],
#                   [gamma*b1/2, -2*np.pi*det]], dtype=complex) * 1/2
    
#     # Calculate unitary propagator
#     U = expm(-1.0j*H*dt)

#     return U

def althardPulsePropagator(b1, g, dt, det):
    gamma = g*muB/hbar
    det = 2*np.pi*det
    w1 = gamma*b1/2 # power of b1 halved by RWA
    weff = np.sqrt(w1**2 + det**2)
    beta = weff*dt
    U = np.array([[np.cos(beta/2) - 1.0j * det/weff * np.sin(beta/2), -1.0j * w1/weff * np.sin(beta/2)],
                  [-1.0j*w1/weff*np.sin(beta/2), np.cos(beta/2) + 1.0j*det/weff * np.sin(beta/2)]])

    return U


def calcPolarisations(states, Pi):
    if Pi == 'Px':
        sigma = np.array([[0, 1],
                          [1, 0]])
    elif Pi == 'Py':
        sigma = np.array([[0, -1.0j],
                          [1.0j, 0]])
    elif Pi == 'Pz':
        sigma = np.array([[1, 0],
                          [0, -1]])
        
    if np.ndim(states) == 2:
        # single state input
        Pi = np.real(np.matmul(np.conjugate(states).T, np.matmul(sigma, states)))
    else:
        # states in an array
        Pi = np.zeros(len(states))
        for s in range(len(states)):
            Pi[s] = np.real(np.matmul(np.conjugate(states[s]).T, np.matmul(sigma, states[s])))

    return Pi


def overallUnitary(w1x, w1y, det, tau):

    dt = tau/len(w1x)
    U_tot = np.eye(2)

    for i in range(len(w1x)):
        U_i = hardPulsePropagator(w1x[i], w1y[i], det, dt)
        U_tot = np.matmul(U_i, U_tot)

    return U_tot

def average_fidelity(U_desired, U_gate):

    F = 1/2
    sigma_list = [sigmax, sigmay, sigmaz]

    for k in range(3):
        m1 = np.matmul(U_desired,  np.matmul(sigma_list[k], np.conj(U_desired.T))) 
        m2 = np.matmul(U_gate, np.matmul(sigma_list[k], np.conj(U_gate.T)))
        F += 1/12 * np.trace(np.matmul(m1, m2))

    return np.real(F)


### Plotting functions

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


cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]

parula_map = LinearSegmentedColormap.from_list('parula', cm_data)