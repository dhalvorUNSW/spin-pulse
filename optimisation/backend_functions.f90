! simulated annealing optimiser module to be called from python
module backend_functions

    ! Required modules
    use iso_c_binding
    use coeffs_to_pulse
    use update_pulse
    use cost_functions
    use sim_functions

    implicit none

contains

    ! evolveState function with C binding for python
    subroutine evolveState_fast(Np, dt, w1x, w1y, det, psi0_real, psi0_imag, &
                               states_real, states_imag) bind(C, name="evolveState_fast")
        
        ! input parameters
        integer(c_int), value, intent(in) :: Np
        real(c_double), value, intent(in) :: dt
        real(c_double), intent(in) :: w1x(Np)
        real(c_double), intent(in) :: w1y(Np)
        real(c_double), intent(in) :: det(Np)
        real(c_double), intent(in) :: psi0_real(2)
        real(c_double), intent(in) :: psi0_imag(2)
        
        ! output parameters
        real(c_double), intent(out) :: states_real(2, Np)
        real(c_double), intent(out) :: states_imag(2, Np)
        
        ! local variables
        complex(8) :: psi(2)
        complex(8) :: U(2,2)
        integer :: i
        
        ! Initialize psi from real and imaginary parts
        psi(1) = complex(psi0_real(1), psi0_imag(1))
        psi(2) = complex(psi0_real(2), psi0_imag(2))
        
        do i = 1, Np
            U = hard_pulse_unitary_xy(w1x(i), w1y(i), dt, det(i))
            psi = matmul(U, psi)
            states_real(1, i) = real(psi(1))
            states_real(2, i) = real(psi(2))
            states_imag(1, i) = aimag(psi(1))
            states_imag(2, i) = aimag(psi(2))
        end do
        
    end subroutine evolveState_fast

    ! gradient ascent pulse tuning subroutine
    subroutine run_grad_ascent(Np, band_dig, amp_dig, amp_max, det_max, w1_max, learning_rate, tau, w1x, best_error) & 
                                bind(C, name="run_grad_ascent")

        ! Input parameters
        integer(c_int), value, intent(in) :: Np
        integer(c_int), value, intent(in) :: band_dig
        integer(c_int), value, intent(in) :: amp_dig
        real(c_double), value, intent(in) :: amp_max
        real(c_double), value, intent(in) :: det_max
        real(c_double), value, intent(in) :: w1_max
        real(c_double), value, intent(in) :: learning_rate
        real(c_double), value, intent(in) :: tau

        ! output params
        real(c_double), intent(out) :: w1x(Np)
        real(c_double), intent(out) :: best_error

        ! Local variables
        complex(8) :: U_list(2, 2, Np), X_list(2, 2, Np), P_list(2, 2, Np)
        complex(8) :: X_prev(2, 2), P_prev(2, 2), Xj(2, 2), Pj(2, 2)
        complex(8) :: U(2, 2), U_desired(2, 2), Hkx(2, 2), prod(2,2), trace
        real(8) :: pass_local_grads(band_dig, Np), grads(Np)
        real(8) :: det, dt, E, peak_val
        integer :: d, j, count

        dt = tau / dble(Np)
        U_desired = reshape([(1.0d0, 0.0d0), (0.0d0, -1.0d0), &
                            (0.0d0, -1.0d0), (1.0d0, 0.0d0)], [2,2])
        U_desired = 1/sqrt(2.0d0) * U_desired

        Hkx = reshape([(0.0d0, 0.0d0), (1.0d0, 0.0d0), &
                        (1.0d0, 0.0d0), (0.0d0, 0.0d0)], [2,2])
        Hkx = 1/2.0d0 * Hkx
        E = 0

        ! Get initial amp_max
        peak_val = abs(maxval(w1x))

        count = 0
        ! do while (best_error < 0.99999999)
        do while (peak_val < w1_max)
            do d=1,band_dig
                det = -det_max + dble(d - 1) *(2 * det_max)/dble(band_dig - 1)
                
                U_list = (0.0d0, 0.0d0)
                X_list = (0.0d0, 0.0d0)
                P_list = (0.0d0, 0.0d0)

                ! Get forward propogated matrices
                X_prev = reshape([(1.0d0, 0.0d0), (0.0d0, 0.0d0), &
                                (0.0d0, 0.0d0), (1.0d0, 0.0d0)], [2,2])

                do j=1,Np
                    U = hard_pulse_unitary(w1x(j), dt, det)
                    U_list(:, :, j) = U
                    Xj = matmul(U, X_prev)
                    X_list(:, :, j) = Xj
                    X_prev = Xj
                end do

                ! Get back propogated matrices
                P_prev = U_desired
                P_list(:, :, Np) = P_prev

                do j=1,Np-1
                    Pj = matmul(conjg(transpose(U_list(:, :, Np - j + 1))), P_prev)
                    P_list(:, :, Np - j) = Pj
                    P_prev = Pj
                end do

                ! Cumulative sum of operator fidelity over detunings
                prod = matmul(conjg(transpose(U_desired)), X_list(:, :, Np))
                trace = prod(1, 1) + prod(2, 2)
                E =  E + 0.25d0 * abs(trace) ** 2
                
                ! Calculate gradients
                grads = 0.0d0
                do j=1,Np
                    Xj = X_list(:, :, j)
                    Pj = P_list(:, :, j)
                    grads(j) = approxGradient(Xj, Pj, Hkx, dt)

                end do

                pass_local_grads(d, :) = grads

            end do

            grads = sum(pass_local_grads, dim=1) / dble(band_dig)
            best_error = E / dble(band_dig)
            E = 0

            ! update pulse shape
            w1x = w1x + learning_rate * grads

            ! print *, "Best error= ", 1 - best_error
            count  = count + 1

            peak_val = abs(maxval(w1x))

        end do

    end subroutine run_grad_ascent

    ! Main simulated annealing function with C binding for Python interoperability
    subroutine run_annealing(Np, n_max, band_dig, amp_dig, amp_max, det_max, &
                            w1_max, lambda, tau, best_sin_coeffs, best_cos_coeffs, best_error) &
                            bind(C, name="run_annealing")

        ! Input parameters
        integer(c_int), value, intent(in) :: Np           ! Pulse length
        integer(c_int), value, intent(in) :: n_max        ! Max Fourier coeffs
        integer(c_int), value, intent(in) :: band_dig     ! Samples per 1/tau in spectrum
        integer(c_int), value, intent(in) :: amp_dig      ! Digits in amplitude
        real(c_double), value, intent(in) :: w1_max       ! Max pulse amplitude
        real(c_double), value, intent(in) :: lambda       ! Penalty parameter
        real(c_double), value, intent(in) :: amp_max      ! Max amplitude
        real(c_double), value, intent(in) :: det_max      ! Max detuning
        real(c_double), value, intent(in) :: tau          ! Time constant
        
        ! Output parameter - changed from function result to output argument
        real(c_double), intent(out) :: best_sin_coeffs(n_max)
        real(c_double), intent(out) :: best_cos_coeffs(n_max + 1)
        real(c_double), intent(out) :: best_error

        ! Local variables
        real(8), parameter :: pi = 4.0d0 * atan(1.0d0)
        real(8) :: w1(Np), w1_new(Np) ! Pulse array
        real(8) :: cos_coeffs(n_max + 1), new_cos_coeffs(n_max + 1), cos_step_sizes(n_max + 1)
        real(8) :: sin_coeffs(n_max), new_sin_coeffs(n_max), sin_step_sizes(n_max)
        real(8) :: E, E_new, dE, E_best
        real(8) :: success_ratio, T, P_acc, R1, cooling_rate
        integer :: up_attempt, up_success, up_attempt_max, up_success_max
        integer :: i, updateX, step
        character(len=100) :: filename

        ! Annealing Parameters
        T = 1 ! Initial temp
        cooling_rate = 0.9
        up_attempt_max = (2*n_max + 1) * 2000
        up_success_max = up_attempt_max / 10
        up_attempt = 0
        up_success = 0
        success_ratio = 1.0d0
        step = 0

        ! Initialise coeffs
        sin_coeffs = 0.0d0
        cos_coeffs = 0.0d0
        cos_coeffs(1) = 0.25d0

        ! Step sizes
        cos_step_sizes(1:2) = 0.1d0
        sin_step_sizes(1) = 0.1d0
        do i = 2,n_max
            cos_step_sizes(i + 1) = cos_step_sizes(i) * 0.5d0
            sin_step_sizes(i) = sin_step_sizes(i - 1) * 0.5d0
        end do

        ! Initialise pulse
        w1 = full_coeffs_to_pulse(cos_coeffs, sin_coeffs, Np, tau)
        E = pass_unitary_error_X2(w1, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max)
        E_best = E
        best_cos_coeffs = cos_coeffs
        best_sin_coeffs = sin_coeffs

        updateX = 1;

        ! do while (success_ratio > 0.01d0)
        do while (success_ratio > 0.1d0)
            ! Propose new coeffs
            if (updateX <= n_max + 1) then
                new_cos_coeffs = update_coeffs(cos_coeffs, cos_step_sizes)
                new_sin_coeffs = sin_coeffs
                updateX = updateX + 1
            else
                new_cos_coeffs = cos_coeffs
                new_sin_coeffs = update_coeffs(sin_coeffs, sin_step_sizes)
                updateX = updateX + 1
                if (updateX > (2*n_max + 1)) then
                    updateX = 1
                end if
            end if
            
            w1_new = full_coeffs_to_pulse(new_cos_coeffs, new_sin_coeffs, Np, tau)

            ! Calculate error of new coeffs
            E_new = pass_unitary_error_X2(w1_new, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max)
            dE = E_new - E
            ! print *, "Error= ", dE

            ! Accept or reject
            if (dE <= 0) then
                cos_coeffs = new_cos_coeffs
                sin_coeffs = new_sin_coeffs
                E = E_new
            else
                up_attempt = up_attempt + 1
                P_acc = dexp(-dE/T)
                call random_number(R1)
                if (P_acc > R1) then
                    up_success = up_success + 1
                    cos_coeffs = new_cos_coeffs
                    sin_coeffs = new_sin_coeffs
                    E = E_new
                end if
            end if

            ! Check if best error has improved
            if (E < E_best) then
                E_best = E
                best_cos_coeffs = cos_coeffs
                best_sin_coeffs = sin_coeffs
                ! print *, "Best error= ", E_best
            end if

            ! Update temperature
            if (up_attempt == up_attempt_max .or. up_success == up_success_max) then
                T = cooling_rate * T
                success_ratio = dble(up_success)/dble(up_attempt)
                print *, "Temperature reduced to ", T
                print *, "Uphill success ratio ", success_ratio
                print *, "Best error= ", E_best
                up_attempt = 0
                up_success = 0

                ! Update log
                ! Open file in append mode
                ! open(unit=99, file="annealing_log.txt", status="unknown", position="append", action="write")
                ! ! Write log message
                ! write(99, '(A,I5,A,F10.4,A,F12.6)') "Step: ", step, " Best energy: ", E_best, "Success ratio: ", success_ratio
                ! ! Close file
                ! close(99)

            end if

            ! Update step
            step = step + 1

        end do

        ! print *, "Done!"
        ! print *, "Best Error= ", E_best

        best_error = E_best

        w1 = full_coeffs_to_pulse(best_cos_coeffs, best_sin_coeffs, Np, tau)

    end subroutine run_annealing

end module backend_functions