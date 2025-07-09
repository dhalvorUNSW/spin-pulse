# class NelderMead:

#     def __init__(self, fourier_pulse, alpha=1, gamma=2, rho=0.5, sigma=0.5):
#         # Metaparameters
#         self.alpha = alpha # Reflection coefficient
#         self.gamma = gamma # Expansion coefficient
#         self.rho = rho # Contraction coefficient
#         self.sigma = sigma # Shrink coefficient

#         if fourier_pulse.time_symmetric == True:
#             self.init_params = fourier_pulse.cos_coeffs # List of starting parameters

#         self.init_pulse = fourier_pulse

#         # Generate initial set of test points 
#         self.generate_test_points()

#     def generate_test_points(self):
#         n = len(self.init_params)
#         self.params = np.zeros([n+1, n])
#         self.params[0, :] = self.init_params
#         for i in range(1, n+1):
#             # Set step size
#             if i <= 2:
#                 step_size = 0.2
#             else:
#                 step_size = step_size * 1 # 0.6
#             # Add set of params
#             R = 2*np.random.rand() - 1
#             self.params[i, :] = self.init_params
#             self.params[i, i-1] = self.params[i, i-1] + R*step_size

#     def run(self, n_steps):
#         # Call error function for all params and order array
#         n = len(self.init_params)
#         N_shots = 300
#         params_with_E = []

#         # Order initial values by E
#         for i in range(n+1):
#             print(f"Evaluating initial param set {i+1}.")
#             test_pulse = self.init_pulse
#             test_pulse.cos_coeffs = self.params[i, :]
#             test_pulse.coeffs_to_pulse()
#             E = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)
#             params_with_E.append((self.params[i, :],  E))
#             print(f"Param set {i+1} error= {E}")

#         # Sort the list based on the cost function output (E)
#         params_with_E.sort(key=lambda x: x[1])  # x[1] is the E value in the tuple
#         sorted_params = np.array([param for param, _ in params_with_E]) # Sorted params
#         sorted_E = np.array([e for _, e in params_with_E])

#         for m in range(n_steps):
#             if m>0:
#                 # Resort arrays
#                 params_with_E = [(sorted_params[_, :], sorted_E[_]) for _ in range(n+1)]
#                 params_with_E.sort(key=lambda x: x[1])  # x[1] is the E value in the tuple
#                 sorted_params = np.array([param for param, _ in params_with_E]) # Sorted params
#                 sorted_E = np.array([e for _, e in params_with_E])

#                 print(f"Best error={sorted_E[0]}")
            
#             # Calculate centroid of params except worst
#             cent = np.mean(sorted_params[:-2, :], 0)

#             # Reflect worst point and evaluate
#             refl = cent + self.alpha * (cent - sorted_params[-1, :])
#             test_pulse.cos_coeffs = refl
#             test_pulse.coeffs_to_pulse()
#             E_refl = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)

#             if E_refl>=sorted_E[0] and E_refl<sorted_E[-2]:
#                 # Replace worst point with reflected point
#                 sorted_E[-1] = E_refl
#                 sorted_params[-1, :] = refl
#                 print("Reflected accepted!")

#             elif E_refl < sorted_E[0]:
#                 # Expand
#                 expd = cent + self.gamma * (refl - cent)
#                 test_pulse.cos_coeffs = expd
#                 test_pulse.coeffs_to_pulse()
#                 E_expd = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)
#                 print("Expand!")
#                 if E_expd < E_refl:
#                     # Replace worst point with expanded point
#                     sorted_E[-1] = E_expd
#                     sorted_params[-1, :] = expd
#                     print("Expanded accepted!")
#                 else:
#                     sorted_E[-1] = E_refl
#                     sorted_params[-1, :] = refl
#                     print("Reflected accepted!")
#             elif E_refl < sorted_E[-1]:
#                 # Contract
#                 cont = cent + self.rho*(refl - cent)
#                 test_pulse.cos_coeffs = cont
#                 test_pulse.coeffs_to_pulse()
#                 E_cont = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)
#                 print("Second worst: Contract!")
#                 if E_cont < E_refl:
#                     sorted_E[-1] = E_cont
#                     sorted_params[-1, :] = cont
#                     print("Contracted point accepted!")
#                 else:
#                     print("We have to shrink!")
#                     for j in range(1, n+1):
#                         sorted_params[j, :] = sorted_params[0, :] + self.sigma * (sorted_params[j, :] - sorted_params[0, :])
#                         test_pulse.cos_coeffs = cont
#                         test_pulse.coeffs_to_pulse()
#                         sorted_E[j] = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)

#             elif E_refl >= sorted_E[-1]:
#                 cont = cent + self.rho*(sorted_params[-1, :] - cent)
#                 test_pulse.cos_coeffs = cont
#                 test_pulse.coeffs_to_pulse()
#                 E_cont = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)
#                 print("Worst: Contract!")
#                 if E_cont < sorted_E[-1]:
#                     sorted_E[-1] = E_cont
#                     sorted_params[-1, :] = cont
#                     print("Contracted point accepted!")
#                 else:
#                     print("We have to shrink!")
#                     for j in range(1, n+1):
#                         sorted_params[j, :] = sorted_params[0, :] + self.sigma * (sorted_params[j, :] - sorted_params[0, :])
#                         test_pulse.cos_coeffs = sorted_params[j, :]
#                         test_pulse.coeffs_to_pulse()
#                         sorted_E[j] = digitalRabiError(test_pulse, N_shots=N_shots, det_noise_width=1e6, amp_noise_width=0.5e6)