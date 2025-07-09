! Module containing cost functions for optimisation of pulse shapes

module cost_functions

    use sim_functions
    use matrix_utils

    implicit none

contains

    function immse(A, B) result(mse)
        implicit none
        real(8), intent(in) :: A(:, :), B(:, :)  ! Input arrays
        integer :: N  ! Number of elements
        real(8) :: mse  ! Mean squared error

        ! Ensure arrays are the same size
        if (size(A) /= size(B)) then
            print *, "Error: Arrays must be the same size!"
            stop
        end if

        N = size(A)  ! Get number of elements
        mse = sum((A - B) ** 2) / dble(N)  ! Compute MSE
    end function immse

    function approxGradient(Xj, Pj, Hk, dt) result(grad)
    
        implicit none
        ! Input variables
        complex(8), intent(in) :: Xj(2, 2), Pj(2, 2), Hk(2, 2)
        real(8), intent(in) :: dt

        ! local variables
        complex(8) :: ket(2, 2), trace1, trace2
        real(8) :: grad

        ket = (0.0d0, 1.0d0) * dt * matmul(Hk, Xj)
        trace1 = trace2x2(matmul(conjg(transpose(Pj)), ket))
        trace2 = trace2x2(matmul(conjg(transpose(Xj)), Pj))
        grad = -0.5d0 * real(trace1 * trace2)

    end function approxGradient

    function xy_unitary_error_X2(w1x, w1y, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max) result(E)
        implicit none
        real(8), intent(in) :: w1x(:), w1y(:)
        real(8), intent(in) :: tau, w1_max, lambda, det_max, amp_max
        integer, intent(in) :: band_dig, amp_dig
        real(8) :: w1(size(w1x)), w1x_scaled(size(w1x)), w1y_scaled(size(w1y)), w1_excess(size(w1x)), amps(amp_dig)
        real(8) :: fids(amp_dig, band_dig), det, E, dt, P
        complex(8) :: U(2,2), C(2,2), prod(2,2), U_desired(2,2), trace
        integer :: i, d, Np, j

        ! Define array of scales
        if (amp_dig == 1) then
            amps(1) = 0
        else
            do j = 1, amp_dig
                amps(j) = -amp_max + (j - 1) * (amp_max + amp_max) / (amp_dig - 1)
            end do
        end if

        ! U_desured
        U_desired = reshape([(1.0d0, 0.0d0), (0.0d0, -1.0d0), &
                            (0.0d0, -1.0d0), (1.0d0, 0.0d0)], [2,2])
        U_desired = 1/sqrt(2.0d0) * U_desired

        ! Compute time step
        Np = size(w1)
        dt = tau / dble(Np)

        do j = 1, amp_dig
            w1x_scaled = w1x * (amps(j) + 1)
            w1y_scaled = w1y * (amps(j) + 1)
            ! Detunings range: -det_max <-> det_max
            do d = 1, band_dig
                det = -det_max + dble(d - 1) *(2 * det_max)/dble(band_dig - 1)
                ! Initial identity operator
                U = (0.0d0, 0.0d0)
                U(1, 1) = (1.0d0, 0.0d0)
                U(2, 2) = (1.0d0, 0.0d0)

                ! Propogate unitary
                do i = 1,Np
                    C = hard_pulse_unitary_xy(w1x_scaled(i), w1y_scaled(i), dt, det)
                    U = matmul(C, U)
                end do
                ! Calculate fidelity of final unitary
                prod = matmul(conjg(transpose(U_desired)), U)
                trace = prod(1, 1) + prod(2, 2)
                fids(j, d) = 0.25d0 * abs(trace) ** 2 
            end do
        end do

        ! Compute w1_excess = max(0, abs(w1) - w1_max)
        w1 = sqrt(w1x**2 + w1y**2)
        do i = 1, (size(w1))
            w1_excess(i) = max(0.0d0, abs(w1(i)) - w1_max)
        end do

        ! Compute penalty P = lambda * sum(w1_excess^2)
        P = lambda * sum(w1_excess**2)

        E = 1 - sum(fids) / (dble(band_dig) * dble(amp_dig)) + P


    end function

    function pass_unitary_error_X2(w1, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max) result(E)
        implicit none
        real(8), intent(in) :: w1(:)
        real(8), intent(in) :: tau, w1_max, lambda, det_max, amp_max
        integer, intent(in) :: band_dig, amp_dig
        real(8) :: w1_scaled(size(w1)), w1_excess(size(w1)), amps(amp_dig)
        real(8) :: fids(amp_dig, band_dig), det, E, dt, P
        complex(8) :: U(2,2), C(2,2), prod(2,2), U_desired(2,2), trace
        integer :: i, d, Np, j

        ! Define array of scales
        if (amp_dig == 1) then
            amps(1) = 0
        else
            do j = 1, amp_dig
                amps(j) = -amp_max + (j - 1) * (amp_max + amp_max) / (amp_dig - 1)
            end do
        end if

        ! U_desured
        U_desired = reshape([(1.0d0, 0.0d0), (0.0d0, -1.0d0), &
                            (0.0d0, -1.0d0), (1.0d0, 0.0d0)], [2,2])
        U_desired = 1/sqrt(2.0d0) * U_desired

        ! Compute time step
        Np = size(w1)
        dt = tau / dble(Np)

        do j = 1, amp_dig
            w1_scaled = w1 * (amps(j) + 1)
            ! Detunings range: -det_max <-> det_max
            do d = 1, band_dig
                det = -det_max + dble(d - 1) *(2 * det_max)/dble(band_dig - 1)
                ! Initial identity operator
                U = (0.0d0, 0.0d0)
                U(1, 1) = (1.0d0, 0.0d0)
                U(2, 2) = (1.0d0, 0.0d0)

                ! Propogate unitary
                do i = 1,Np
                    C = hard_pulse_unitary(w1_scaled(i), dt, det)
                    U = matmul(C, U)
                end do
                ! Calculate fidelity of final unitary
                prod = matmul(conjg(transpose(U_desired)), U)
                trace = prod(1, 1) + prod(2, 2)
                fids(j, d) = 0.25d0 * abs(trace) ** 2 ! 
            end do
        end do

        ! Compute w1_excess = max(0, abs(w1) - w1_max)
        do i = 1, (size(w1))
            w1_excess(i) = max(0.0d0, abs(w1(i)) - w1_max)
        end do

        ! Compute penalty P = lambda * sum(w1_excess^2)
        P = lambda * sum(w1_excess**2)

        E = 1 - sum(fids) / (dble(band_dig) * dble(amp_dig)) + P


    end function

    function pass_proj_error_X2(w1, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max) result(E)
        implicit none
        real(8), intent(in) :: w1(:)
        real(8), intent(in) :: tau, w1_max, lambda, amp_max, det_max
        integer, intent(in) :: band_dig, amp_dig
        real(8) :: Px(amp_dig, band_dig), Py(amp_dig, band_dig), Pz(amp_dig, band_dig)
        real(8) :: w1_scaled(size(w1)), w1_excess(size(w1)), amps(amp_dig)
        real(8) :: Px_ideal(amp_dig, band_dig), Py_ideal(amp_dig, band_dig), Pz_ideal(amp_dig, band_dig)
        complex(8) :: psi0(2), final_state(2)
        real(8) :: det, E, P
        integer :: d, i

        ! Define psi0
        psi0(1) = (1.0d0, 0.0d0)
        psi0(2) = (0.0d0, 0.0d0)

        ! Define ideal polarisations
        Px_ideal = 0.0d0
        Py_ideal = -1.0d0
        Pz_ideal = 0.0d0

        ! Define array of scales
        if (amp_dig == 1) then
            amps(1) = 0
        else
            do i = 1, amp_dig
                amps(i) = -amp_max + (i - 1) * (amp_max + amp_max) / (amp_dig - 1)
            end do
        end if

        do i = 1, amp_dig
            w1_scaled = w1 * (amps(i) + 1)
            do d = 1, band_dig
                det = (det_max) * (d - 1) / dble(band_dig - 1)
                final_state = get_final_state(psi0, w1_scaled, tau, det)
                Px(i, d) = polarisation(final_state, 1)
                Py(i, d) = polarisation(final_state, 2)
                Pz(i, d) = polarisation(final_state, 3)
            end do
        end do

        E = immse(Px, Px_ideal) + immse(Py, Py_ideal) + immse(Pz, Pz_ideal)

        ! Compute w1_excess = max(0, abs(w1) - w1_max)
        do i = 1, (size(w1))
            w1_excess(i) = max(0.0d0, abs(w1(i)) - w1_max)
        end do

        ! Compute penalty P = lambda * sum(w1_excess^2)
        P = lambda * sum(w1_excess**2)

        E = E + P

    end function


end module cost_functions