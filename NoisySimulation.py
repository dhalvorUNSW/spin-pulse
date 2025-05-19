#%%
import numpy as np
from numpy import linalg as lin
import cmath, math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from scipy import linalg
from scipy.fft import fft, fftfreq
from scipy.io import savemat, loadmat
from scipy.linalg import expm, sinm, cosm
from IPython import display

#%% Qubit parameters
gamma_e = 28e9

# Static field (T)
B0 = 1

# Driving field (T)
B1 = 100/28*1e-3#10/28*1e-3#

# Larmour frequency (Hz)
omega = 2*np.pi*gamma_e*B0

# Rabi frequency (Hz)
omega1 = 2*np.pi*gamma_e*B1
f1 = omega1 / (2*np.pi)

up = np.array([[1], [0]])
down = np.array([[0], [1]])

#%% Functions
# For plotting Bloch spheres
def make_3d_circle(radius=1, center=(0, 0, 0), normal=(0, 0, 1), num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)
    circle_z = np.zeros_like(circle_x)
    
    normal = np.array(normal) / np.linalg.norm(normal)  # Normalize the normal vector
    
    # Find a vector perpendicular to the normal
    if normal[0] == 0 and normal[1] == 0:
        perp = np.array([1, 0, 0])
    else:
        perp = np.array([-normal[1], normal[0], 0])
    # perp = np.linalg.norm(perp)
    
    # Find a second perpendicular vector
    perp2 = np.cross(normal, perp)
    
    # Rotate the circle points into the plane
    x_3d = center[0] + circle_x * perp[0] + circle_y * perp2[0]
    y_3d = center[1] + circle_x * perp[1] + circle_y * perp2[1]
    z_3d = center[2] + circle_x * perp[2] + circle_y * perp2[2]
    
    return x_3d, y_3d, z_3d

# Generate noise
def generate_broadband_noise(num_samples=1024, exponent=0):
    """
    Generate noise with a specific frequency spectrum.
    
    Parameters:
    - num_samples: Number of points in the signal.
    - exponent: Controls the power spectral density (PSD) slope.
      - 0  -> White noise (flat spectrum)
      - 1 -> Pink noise (1/f spectrum)
      - 2 -> Brown noise (1/f^2 spectrum)
    
    Returns:
    - noise: Time-domain noise signal.
    """
    # Generate random Fourier coefficients
    freqs = np.fft.rfftfreq(num_samples)  # Positive frequency components
    amplitudes = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))  # Complex noise

    # Apply frequency-dependent scaling
    with np.errstate(divide='ignore', invalid='ignore'):
        scaling = np.where(freqs == 0, 0, 1 / (freqs ** (exponent / 2.0)))  # PSD ~ 1/f^exponent
        amplitudes *= scaling

    # Convert back to time domain
    noise = np.fft.irfft(amplitudes, num_samples)

    # Normalize to unit variance
    noise /= np.std(noise)

    return noise

def generate_two_level_noise(duration, switching_rate, level1=-1, level2=1, sample_rate=1000):
    """
    Generate two-level fluctuation noise with a given switching rate.

    Parameters:
        duration (float): Total duration of the signal in seconds.
        switching_rate (float): Average number of switches per second.
        level1 (float): First noise level.
        level2 (float): Second noise level.
        sample_rate (int): Sampling rate in Hz.

    Returns:
        t (numpy array): Time vector.
        signal (numpy array): Generated noise signal.
    """
    
    num_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, num_samples)
    signal = np.zeros(num_samples)
    
    current_level = np.random.choice([level1, level2])
    signal[0] = current_level
    
    # Generate switching times using a Poisson process
    switch_prob = switching_rate / sample_rate
    
    for i in range(1, num_samples):
        if np.random.rand() < switch_prob:
            current_level = level1 if current_level == level2 else level2
        signal[i] = current_level
    
    return signal

# Noisy FID generator
def Ramsey_Single_Shot(omega=omega, gamma=gamma_e, B1=B1, dt=0.5e-9,
                       t_X2=0.25*1/f1, t_wait=5e-9,
                       delta_0=0.25*f1,
                       delta_f=np.zeros(2*int(0.25*1/f1/0.5e-9)+int(5e-9/0.5e-9)),
                       psi0=down, plot_trajectory=False):
    # Pauli matrices
    sigma_x = 0.5*np.array([[0, 1], [1, 0]])
    sigma_y = 0.5*np.array([[0, -1j], [1j, 0]])
    sigma_z = 0.5*np.array([[1, 0], [0, -1]])
    
    # Time array (s)
    t_max = t_X2 + t_wait + t_X2
    
    npts_X2 = int(t_X2/dt)
    npts_wait = int(t_wait/dt)
    npts_max = npts_X2 + npts_wait + npts_X2
    
    # Non-zero time
    if (npts_X2>=1) and (npts_wait>=1):
    
        time = np.linspace(0, t_max, npts_max)
        
        # fLarmor detuning (Hz)
        # delta = np.array([0]*npts_X2 + [delta_0]*npts_wait + [0]*npts_X2)
        delta = np.array([delta_0]*npts_X2 + [delta_0]*npts_wait + [delta_0]*npts_X2)\
            + delta_f
        
        # Driving term
        driving = np.array([np.pi*gamma*B1]*npts_X2 +
                           [0]*npts_wait +
                           [np.pi*gamma*B1]*npts_X2)
        
        # State vectors in rotating frame
        psi_rot_t = np.zeros((2, len(time)), dtype = np.complex64)
        psi_rot_t[:,0] = psi0[:,0]
        
        # State vectors in lab frame
        psi_lab_t = np.zeros((2, len(time)), dtype = np.complex64)
        psi_lab_t[:,0] = psi0[:, 0]
        
        # Time evolution in rotating frame
        for i in range(1, npts_max): 
            H = np.array([[-delta[i],           driving[i]],
                          [np.conj(driving[i]), delta[i] ]])
            Udt = expm(-1j*H*dt)
            psi_rot_t[:, i] = Udt @ psi_rot_t[:, i-1]
        
        # Expectation values & spin-up probability
        expec_rot_x = np.zeros((len(time)))
        expec_rot_y = np.zeros((len(time)))
        expec_rot_z = np.zeros((len(time)))
        P_z = np.zeros((len(time)))
        
        expec_lab_x = np.zeros((len(time)))
        expec_lab_y = np.zeros((len(time)))
        expec_lab_z = np.zeros((len(time)))
        
        for i in range(0, len(time)):
            expec_rot_x[i] = np.real(np.conj(psi_rot_t[:,i]) @ (sigma_x) @ (psi_rot_t[:,i]))
            expec_rot_y[i] = np.real(np.conj(psi_rot_t[:,i]) @ (sigma_y) @ (psi_rot_t[:,i]))
            expec_rot_z[i] = np.real(np.conj(psi_rot_t[:,i]) @ (sigma_z) @ (psi_rot_t[:,i]))
            P_z[i] = np.abs(np.conj(up.T) @ psi_rot_t[:,i]) ** 2
        
            # Calculate the expectation values along the x,y,z axis when using the passage matrix
            # to transform from the rotating frame to the lab frame
            
            passage_matrix = [[np.exp(-1j*omega*time[i]/2), 0                           ], 
                              [0,                           np.exp((1j*omega*time[i])/2)]]
            psi_lab_t[:,i] = passage_matrix @ psi_rot_t[:,i]
            P_z[i] = np.abs(np.conj(up.T) @ psi_lab_t[:,i]) ** 2
            expec_lab_x[i] = np.real(np.conj(psi_lab_t[:,i].T) @ sigma_x @ psi_lab_t[:,i])
            expec_lab_y[i] = np.real(np.conj(psi_lab_t[:,i].T) @ sigma_y @ psi_lab_t[:,i])
            expec_lab_z[i] = np.real(np.conj(psi_lab_t[:,i].T) @ sigma_z @ psi_lab_t[:,i])
        
        P_x = expec_rot_x[-1] + 0.5
        P_y = expec_rot_y[-1] + 0.5
        P_z = expec_rot_z[-1] + 0.5
          
    # Zero time
    else:
        P_x = np.real(np.conj(psi0[:,0]) @ (sigma_x) @ (psi0[:,0])) + 0.5
        P_y = np.real(np.conj(psi0[:,0]) @ (sigma_y) @ (psi0[:,0])) + 0.5
        P_z = np.real(np.conj(psi0[:,0]) @ (sigma_z) @ (psi0[:,0])) + 0.5

    return P_x, P_y, P_z

#%% Generate and plot noise
num_samples = 1000
broadband_noise = generate_broadband_noise(num_samples, exponent=2)

duration = 1/1e9  # seconds
switching_rate = 100*1e9  # switches per second
sample_rate = 1000*1e9  # Hz
two_level_noise = generate_two_level_noise(duration, switching_rate, sample_rate=sample_rate)

noise = broadband_noise# + 2.5*two_level_noise

# # Plot the generated two-level noise
# plt.figure(figsize=(10, 4))
# plt.plot(two_level_noise)
# plt.title("Two-level fluctuation noise")
# plt.xlabel("Sample Index")
# plt.ylabel("Amplitude")
# plt.ylim(-1.5, 1.5)
# plt.grid()
# plt.show()

linewidth = 2
plt.rcParams.update({'font.size': 11})

fig, axs = plt.subplots(2)

axs[0].plot(noise-noise[0], color="hotpink", linewidth=linewidth)
axs[0].set_xlabel("Sample index")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Time domain")
axs[0].grid()

psd = np.abs(np.fft.rfft(noise)) ** 2
freqs = np.fft.rfftfreq(num_samples)

axs[1].loglog(freqs[1:], psd[1:], color="hotpink")
axs[1].set_xlabel("Frequency")
axs[1].set_ylabel("Power spectral density")
axs[1].set_title("Power spectral density of different noise types")
axs[1].grid()

plt.tight_layout()
plt.show()

#%% Single shot
# P_x, P_y, P_z = Ramsey_Single_Shot(plot_trajectory=True)

#%% Ramsey oscillations
t_X2 = 0.25*1/f1

delta_0_min = -10*f1
delta_0_max = 10*f1
dt = 0.5e-9#1e-9#
t_wait_min = dt
t_wait_max = 100e-9#20e-9

npts_delta_0_min_max = 100#50#
npts_wait_min_max = 100#50#
n_shots = 100#50#

# Broadband noise parameter
broadband_noise_amp = 0.5*f1 # 0.05*f1 # 

# Two-level noise parameters
switching_rate = 1e4 # switches per second
two_level_noise_amp = 0.5*f1
sine_noise_amp = 0.5*f1
sine_noise_freq = 1e3

delta_0_array = np.linspace(delta_0_min, delta_0_max, npts_delta_0_min_max)
t_wait_array = np.linspace(t_wait_min, t_wait_max, npts_wait_min_max)

t_total = ( np.sum(t_wait_array) + len(t_wait_array)*2*t_X2 ) * npts_delta_0_min_max * n_shots
npts_total = int( ( np.sum(t_wait_array) + len(t_wait_array)*2*t_X2 ) * npts_delta_0_min_max * n_shots / dt )

broadband_noise_total = broadband_noise_amp*generate_broadband_noise(npts_total, exponent=1)

two_level_noise_total = two_level_noise_amp*generate_two_level_noise(duration=t_total, switching_rate=switching_rate, sample_rate=1/dt)
# two_level_noise_total = two_level_noise_total - two_level_noise_total[0]

sine_noise_total = sine_noise_amp*np.sin(2*np.pi*sine_noise_freq*np.linspace(0, t_total, num=npts_total))

delta_f_all = np.array([])
npts_counter = 0

live_plotting = False

P_z_all_3D = []

if live_plotting:
    fig, axs = plt.subplots()
    plt.show()

for i in range(npts_wait_min_max):
    P_z_all_2D = []
    for j in range(npts_delta_0_min_max):
        P_z_all_1D = []
        for k in range(n_shots):
            # Two-level switching noise
            delta_f = broadband_noise_total[npts_counter:npts_counter + 2*int(t_X2/dt)+int(t_wait_array[i]/dt)]
            # delta_f = broadband_noise_amp*generate_broadband_noise( 2*int(t_X2/dt)+int(t_wait_array[i]/dt),
            #                               exponent=1 ) +\
            #           two_level_noise_total[npts_counter:npts_counter + 2*int(t_X2/dt)+int(t_wait_array[i]/dt)]
            # Excited state
            # delta_f = broadband_noise_amp*generate_broadband_noise( 2*int(t_X2/dt)+int(t_wait_array[i]/dt),
            #                               exponent=1 ) +\
            #           two_level_noise_amp*np.random.choice([-1, 1])*np.ones(2*int(t_X2/dt)+int(t_wait_array[i]/dt))
            # Sine noise
            # delta_f = broadband_noise_amp*generate_broadband_noise( 2*int(t_X2/dt)+int(t_wait_array[i]/dt),
            #                               exponent=1 ) +\
            #           sine_noise_total[npts_counter:npts_counter + 2*int(t_X2/dt)+int(t_wait_array[i]/dt)]
            P_x, P_y, P_z = Ramsey_Single_Shot(omega=omega, gamma=gamma_e, B1=B1, dt=dt,
                                    t_X2=t_X2, t_wait=t_wait_array[i],
                                    delta_0=delta_0_array[j], delta_f=delta_f,
                                    psi0=down, plot_trajectory=False)
            P_z_all_1D.append(P_z)
            delta_f_all = np.concatenate((delta_f_all, delta_f))
            npts_counter = npts_counter + 2*int(t_X2/dt)+int(t_wait_array[i]/dt)
            # print(npts_counter)
        P_z_all_2D.append(P_z_all_1D)
        print("i="+str(i)+", j="+str(j))
    P_z_all_3D.append(P_z_all_2D)
    # print(i)
    P_z_all_3D_array = np.array(P_z_all_3D)
    P_z_all_2D_averaged_array = np.mean(P_z_all_3D_array, axis=2)
    if (i>1) and (live_plotting==True):
        axs.clear()
        axs.pcolor(delta_0_array/1e6, t_wait_array[0:i+1]/1e-9, P_z_all_2D_averaged_array)
        axs.set_ylabel('Wait time (ns)')
        axs.set_xlabel('Frequency detuning (MHz)')
        plt.tight_layout()
        plt.pause(1)
        # fig.canvas.draw()

if live_plotting:
    axs.set_ylabel('Wait time (ns)')
    axs.set_xlabel('Frequency detuning (MHz)')
    plt.tight_layout()
    plt.show()

P_z_all_3D_array = np.array(P_z_all_3D)
P_z_all_2D_averaged_array = np.mean(P_z_all_3D_array, axis=2)

#%% Plot results
fig, axs = plt.subplots()
axs.pcolor(delta_0_array/1e6, t_wait_array/1e-9, P_z_all_2D_averaged_array)
axs.set_ylabel('Wait time (ns)')
axs.set_xlabel('Frequency detuning (MHz)')
axs.set_xlim([-1e3, 1e3])
plt.tight_layout()
plt.show()

ind_skip = 1

fft_mat = np.empty(np.shape(P_z_all_2D_averaged_array), dtype=complex)[:,0:npts_wait_min_max//2]

n = 0
for t_line in np.transpose(P_z_all_2D_averaged_array):
    f_line = fft(t_line)
    f = fftfreq(npts_wait_min_max, dt)[:npts_wait_min_max//2]
    # plt.plot(f/1e6, 2.0/npts_wait_min_max * np.abs(f_line[0:npts_wait_min_max//2]))
    fft_mat[n] = f_line[0:npts_wait_min_max//2]
    n = n + 1

plt.xlabel('f (MHz)')
plt.ylabel('Wait time (ns)')
plt.tight_layout()
plt.show()

fft_mag = np.abs(fft_mat[:,ind_skip:])
fft_real = np.real(fft_mat[:,ind_skip:])
fft_phs = np.angle(fft_mat[:,ind_skip:])/np.pi

plt.figure()
# plt.pcolor(delta_0_array*1e-6,f[ind_skip:]*1e-6,\
#             np.transpose(2/npts_wait_min_max*fft_real[:,ind_skip:]), shading='auto')
plt.pcolor(delta_0_array*1e-6, f[ind_skip:]*1e-6, np.transpose(fft_mag), shading='auto')
plt.colorbar()
plt.title('FFT magnitude')
plt.xlabel('dfESR (MHz)')
plt.ylabel('f (MHz)')
# plt.clim([-5,10])
plt.tight_layout()
plt.show()

# fig, axs = plt.subplots()
# axs.plot(t_wait_array/1e-9, P_z_all_2D_averaged_array[:, int(0.3*npts_delta_0_min_max)])
# axs.set_xlabel('Wait time (ns)')
# axs.set_ylabel('Pz')
# plt.tight_layout()
# plt.show()

#%% Save results
result_dic = {"t_wait_array": t_wait_array,
              "delta_0_array": delta_0_array,
              "P_z_all_2D_averaged_array": P_z_all_2D_averaged_array,
              "broadband_noise_total": broadband_noise_total,
              }
parent_folder = "C:/Users/labadmin/Desktop/Jonathan_Huang/Frequency_Feedback_Simulation_Hamfleet19/"
savemat(parent_folder+"Ramsey_Simulation_Simple_50MHz_Noise_2p5_ns_Rabi.mat", result_dic)