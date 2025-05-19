import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sigpy.mri.rf import slr

### Set constants used below

hbar = c.hbar
muB = c.value('Bohr magneton')


### Pulse generating functions
def generateBURPPulse(
    g,
    tau,
    Np, 
    bw,
    pulse_type,
    paper=True,
    plot_output=True
):
    if paper == True:
        A_coeffs, B_coeffs = get_paper_BURP_coeffs(pulse_type, 2)
    else:
        A_coeffs = np.array([-0.7339,   -0.3593 ,   0.5101 ,   0.4689 ,  -0.0245])
        B_coeffs = np.array([1.1312  ,  0.7096 ,  -0.3640 ,  -0.2139])

    times = np.linspace(0, tau, Np)
    w1 = coeffs_to_BURP(A_coeffs, B_coeffs, times, tau)
    gamma = g*muB/hbar 
    b1 = w1/gamma

    if plot_output == True:
        C5 = '#360568' # Purple
        fig = plt.figure(figsize=(7, 4))
        plt.plot(times, b1, color=C5)
        plt.xlabel('time (s)')
        plt.ylabel('B1 amplitude (T)')
        plt.title(f'BURP {pulse_type} pulse')

    return b1


def generateSLRPulse(
    tau,
    sr,
    bw,
    pulse_type,
    filter_type='min',
    two_axis_pulse=False,
    plot_output=True
):
    g = 2
    gamma = g*muB/hbar
    n = int(tau*sr) 
    dt = tau/n
    TBW = tau*bw
    scale = 1/(gamma*dt)*2

    if pulse_type == 'pi':
        ptype = 'inv'
    elif pulse_type == 'pi/2':
        ptype = 'ex'
    else:
        print('Pulse type not possible by SLR. Set either pi or pi/2')

    rf = scale * slr.dzrf(n, TBW, ptype=ptype, ftype=filter_type, d1=0.00001, d2=0.00001, cancel_alpha_phs= not two_axis_pulse)
    w1 = np.real(rf)
    w1 = gamma*w1/2

    if plot_output == True:
        times = np.linspace(0, tau, n)
        C5 = '#360568' # Purple
        fig = plt.figure(figsize=(7, 4))
        plt.plot(times, w1, color=C5)
        plt.xlabel('time (s)')
        plt.ylabel('w1 amplitude (Hz)')
        plt.title(f'SLR {pulse_type} pulse')

    return w1

def coeffs_to_BURP(
    A_coeffs,
    B_coeffs,
    t,
    tau,
):  
    w = 2*np.pi/tau
    w1 = A_coeffs[0]
    for i in range(len(B_coeffs)):
        w1 += A_coeffs[i+1]*np.cos((i+1)*w*t)
        w1 += B_coeffs[i]*np.sin((i+1)*w*t)
    
    return w1 * w

def get_paper_BURP_coeffs(
    pulse_type,
    version,
    n_max=None
):
    if pulse_type == 'pi' and version == 1:
        A_coeffs = np.array([0.5, 0.74, -0.2, -0.92, 0.12, -0.03, -0.04, 0.01, -0.02, -0.01])
        B_coeffs = np.array([-1.52, 1.00, -0.31, -0.03, 0.08, -0.05, 0.00, 0.01, -0.01])
        
    elif pulse_type == 'pi' and version == 2:
        A_coeffs = np.array([0.5, 0.79, 0.00, -1.23, -0.19, 0.1, 0.12, 0.04, -0.03, -0.03, -0.01, 0.00])
        B_coeffs = np.array([-0.71, -1.39, 0.31, 0.47, 0.22, 0.03, -0.05, -0.04, 0.00, 0.02, 0.01])

    elif pulse_type == 'pi/2' and version == 1:
        A_coeffs = np.array([0.23, 0.89, -1.02, -0.25, 0.14, 0.03, 0.04, -0.03, 0.00])
        B_coeffs = np.array([-0.4, -1.42, 0.74, 0.06, 0.03, -0.04, -0.02, 0.01])

    elif pulse_type == 'pi/2' and version == 2:
        A_coeffs = np.array([0.26, 0.91, 0.45, -1.31, -0.12, 0.03, 0.01, 0.06, 0.01, -0.02, -0.01])
        B_coeffs = np.array([-0.12, -1.79, 0.01, 0.41, 0.08, 0.07, 0.01, -0.04, -0.01, 0.00])

    else:
        print('Error: invalid pulse type.')

    return A_coeffs, B_coeffs