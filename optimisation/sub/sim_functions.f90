! Module containing functions to simulate the time evolution of a given pulse

module sim_functions

    use matrix_utils
    implicit none

contains

    function hard_pulse_unitary(w1, dt, det) result(U)
        implicit none
        real(8), intent(in) :: w1, dt, det
        real(8) :: weff, beta
        complex(8) :: U(2,2)  ! 2x2 complex matrix
        real(8), parameter :: pi = 4.0d0 * atan(1.0d0)

        ! Compute parameters
        weff = sqrt(w1**2 + (2.0d0 * pi * det)**2)
        beta = weff * dt

        ! Construct unitary propagator U
        U(1,1) = dcos(beta/2.0d0) - (0.0d0,1.0d0) * (2.0d0 * pi * det) / weff * dsin(beta/2.0d0)
        U(1,2) = -(0.0d0,1.0d0) * w1 / weff * dsin(beta/2.0d0)
        U(2,1) = -(0.0d0,1.0d0) * w1 / weff * dsin(beta/2.0d0)
        U(2,2) = dcos(beta/2.0d0) + (0.0d0,1.0d0) * (2.0d0 * pi * det) / weff * dsin(beta/2.0d0)

    end function

    function hard_pulse_unitary_xy(w1x, w1y, dt, det) result(U)
        implicit none
        real(8), intent(in) :: w1x, w1y, dt, det
        real(8) :: weff, beta
        complex(8) :: U(2,2)  ! 2x2 complex matrix
        real(8), parameter :: pi = 4.0d0 * atan(1.0d0)

        ! Compute parameters
        weff = sqrt(w1x**2 + w1y**2 + (2.0d0 * pi * det)**2)
        beta = weff * dt

        if (weff == 0.0d0) then
            U(1,1) = (1.0d0, 0.0d0)
            U(1,2) = (0.0d0, 0.0d0)
            U(2,1) = (0.0d0, 0.0d0)
            U(2,2) = (1.0d0, 0.0d0)
        else
            ! Construct unitary propagator U
            U(1,1) = dcos(beta/2.0d0) - (0.0d0,1.0d0) * (2.0d0 * pi * det) / weff * dsin(beta/2.0d0)
            U(1,2) = -(0.0d0,1.0d0) * w1x / weff * dsin(beta/2.0d0) - (1.0d0,0.0d0) * w1y / weff * dsin(beta/2.0d0)
            U(2,1) = -(0.0d0,1.0d0) * w1x / weff * dsin(beta/2.0d0) + (1.0d0,0.0d0) * w1y / weff * dsin(beta/2.0d0)
            U(2,2) = dcos(beta/2.0d0) + (0.0d0,1.0d0) * (2.0d0 * pi * det) / weff * dsin(beta/2.0d0)
        end if

    end function 

    function get_final_state(psi0, w1, tau, det) result(final_state)
        implicit none
        real(8), intent(in) :: tau, det
        real(8), intent(in) :: w1(:)  ! Array of control amplitudes
        complex(8), intent(in) :: psi0(2)  ! Initial qubit state
        complex(8) :: final_state(2)  ! Final state after propagation
        complex(8) :: U(2,2)  ! 2x2 unitary matrix
        real(8) :: dt
        integer :: i, n

        ! Compute time step
        n = size(w1)
        dt = tau / dble(n)

        ! Initialize state
        final_state = psi0

        ! Propagate state through unitary transformations
        do i = 1, n
            U = hard_pulse_unitary(w1(i), dt, det)  ! Compute unitary
            final_state = matmul(U, final_state)  ! Apply unitary to state
        end do

    end function

    function polarisation(state, Pi) result(P)
        implicit none
        integer, intent(in) :: Pi  ! 1 = Px, 2 = Py, 3 = Pz
        complex(8), intent(inout) :: state(2)  ! Input qubit state
        complex(8) :: sigma(2,2), state_conj(2)  ! Pauli matrix
        real(8) :: P  ! Polarization result

        ! Select appropriate Pauli matrix
        select case (Pi)
            case (1)  ! Px
                sigma = reshape([(0.0d0, 0.0d0), (1.0d0, 0.0d0), &
                                (1.0d0, 0.0d0), (0.0d0, 0.0d0)], [2,2])
            case (2)  ! Py
                sigma = reshape([(0.0d0, 0.0d0), (0.0d0, -1.0d0), &
                                (0.0d0, 1.0d0), (0.0d0, 0.0d0)], [2,2])
            case (3)  ! Pz
                sigma = reshape([(1.0d0, 0.0d0), (0.0d0, 0.0d0), &
                                (0.0d0, 0.0d0), (-1.0d0, 0.0d0)], [2,2])
            case default
                print *, "Error: Invalid Pi value. Use 1 (Px), 2 (Py), or 3 (Pz)."
                stop
        end select

        ! Compute P = real(state' * (sigma * state))
        state_conj = conjg(state)
        state = matmul(sigma, state)
        P = real(state_conj(1) * state(1) + state_conj(2) * state(2))

    end function polarisation

end module sim_functions