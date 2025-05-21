import numpy as np
from operators import sigmax, sigmay, sigmaz

class SpinQubit:

    def __init__(self, initial_state="up"):
        """
        Initialise a SpinQubit instance.

        Args:
            initial_state: Initial state of qubit, a 2x1 complex vector.
        """
        self.set_initial_state(initial_state) # Set inital state of qubit


    def set_initial_state(self, initial_state):
        """
        Set inital state of qubit.
        
        Returns:
            Complex vector corresponding to "up" or "down" state.
        """
        if initial_state == "up":
            self.state = np.array([[1], [0]], dtype=complex)
        elif initial_state == "down":
            self.state = np.array([[0], [1]], dtype=complex)
        else:
            raise ValueError("Initial state unrecognised: choose 'up' or 'down'.")

    def rwUnitary(self, w1x, w1y, det, dt):

        det = 2*np.pi*det # Convert to units of radians
        weff = np.sqrt(w1x**2 + w1y**2 + det**2)
        beta = weff*dt # Rotation angle
        if weff == 0:
            U = np.eye(2, dtype=complex)
        else:
            U = np.cos(beta/2)*np.eye(2, dtype=complex)\
                  - 1.0j*np.sin(beta/2)*(w1x/weff*sigmax() + w1y/weff*sigmay() + det/weff*sigmaz())

        return U
    
    # TODO: Lab frame unitary
        