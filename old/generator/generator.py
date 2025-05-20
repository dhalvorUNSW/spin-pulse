import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt
from scipy.linalg import expm
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