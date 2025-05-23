! simulated annealing optimiser module to be called from python
module py_annealing_optimiser

    ! Required modules
    use iso_c_binding ! For python wrapper
    use coeffs_to_pulse
    use update_pulse
    use cost_functions
    use sim_functions

    implicit none

contains

    ! Main simulated annealing function with C binding for Python interoperability
    subroutine run_annealing(Np, n_max, band_dig, amp_dig, amp_max, det_max, &
                            w1_max, lambda, tau, init_coeffs, &
                            best_coeffs, best_error) &
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
        real(c_double), intent(in) :: init_coeffs(n_max)  ! Initial coefficients

        ! Output parameter - changed from function result to output argument
        real(c_double), intent(out) :: best_coeffs(n_max)
        real(c_double), intent(out) :: best_error

        ! Local variables
        real(8) :: w1(Np), w1_new(Np) ! Pulse array
        real(8) :: cos_coeffs(n_max), new_coeffs(n_max), step_sizes(n_max) ! Array of Fourier coeffs
        real(8) :: E, E_new, dE, E_best
        real(8) :: success_ratio, T, P_acc, R1, cooling_rate
        integer :: up_attempt, up_success, up_attempt_max, up_success_max
        integer :: i
        real(8), parameter :: pi = 4.0d0 * atan(1.0d0)

        ! Initialize annealing parameters
        T = 1
        cooling_rate = 0.95
        up_attempt_max = (2*n_max + 1) * 1000
        up_success_max = up_attempt_max / 10
        up_attempt = 0
        up_success = 0
        success_ratio = 1.0d0
        
        ! Initialize coefficients from input
        cos_coeffs = init_coeffs
        
        ! Calculate step sizes
        step_sizes(1:2) = 0.2d0
        do i = 2,(n_max - 1)
            step_sizes(i + 1) = step_sizes(i) * 0.8d0
        end do
        
        ! Initialize pulse and calculate initial error
        w1 = cos_coeffs_to_pulse(cos_coeffs, Np, tau)
        E = pass_proj_error_X2(w1, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max)
        E_best = E
        best_coeffs = cos_coeffs
        
        ! Main annealing loop
        do while (success_ratio > 0.01d0)
            ! Propose new coefficients
            new_coeffs = update_coeffs(cos_coeffs, step_sizes)
            w1_new = cos_coeffs_to_pulse(new_coeffs, Np, tau)
            
            ! Calculate error of new coeffs
            E_new = pass_proj_error_X2(w1_new, w1_max, amp_max, tau, lambda, band_dig, amp_dig, det_max)
            dE = E_new - E
            
            ! Accept or reject
            if (dE <= 0) then
                cos_coeffs = new_coeffs
                E = E_new
            else
                up_attempt = up_attempt + 1
                P_acc = exp(-dE/T)
                
                call random_number(R1)
                if (P_acc > R1) then
                    up_success = up_success + 1
                    cos_coeffs = new_coeffs
                    E = E_new
                end if
            end if
            
            ! Check if best error has improved
            if (E < E_best) then
                E_best = E
                best_coeffs = cos_coeffs
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
            end if
            
        end do

        best_error = E_best

    end subroutine run_annealing

end module py_annealing_optimiser