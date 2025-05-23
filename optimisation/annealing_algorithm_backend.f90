! simulated annealing optimiser module to be called from python
module annealing_algorithm_backend

    ! Required modules
    use iso_c_binding
    use coeffs_to_pulse
    use update_pulse
    use cost_functions
    use sim_functions

    implicit none

contains

    ! Main simulated annealing function with C binding for Python interoperability
    subroutine run_annealing(Np, n_max, band_dig, amp_dig, amp_max, det_max, &
                            init_step, w1_max, lambda, tau, best_sin_coeffs, best_cos_coeffs, best_error) &
                            bind(C, name="run_annealing")

    ! Input parameters
    integer(c_int), value, intent(in) :: Np           ! Pulse length
    integer(c_int), value, intent(in) :: n_max        ! Max Fourier coeffs
    integer(c_int), value, intent(in) :: band_dig     ! Samples per 1/tau in spectrum
    integer(c_int), value, intent(in) :: amp_dig      ! Digits in amplitude
    real(c_double), value, intent(in) :: init_step    ! Initial step size for fourier coeffs
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
    integer :: i, updateX

    ! Annealing Parameters
    T = 1 ! Initial temp
    cooling_rate = 0.95
    up_attempt_max = (2*n_max + 1) * 2000
    up_success_max = up_attempt_max / 10
    up_attempt = 0
    up_success = 0
    success_ratio = 1.0d0

    ! Initialise coeffs
    sin_coeffs = 0.0d0
    cos_coeffs = 0.0d0
    cos_coeffs(1) = 0.25d0

    ! Step sizes
    cos_step_sizes(1:2) = init_step
    sin_step_sizes(1) = init_step
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

    ! Annealing loop:

    updateX = 1;
    do while (success_ratio > 0.01d0)
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
            ! print *, "Temperature reduced to ", T
            ! print *, "Uphill success ratio ", success_ratio
            ! print *, "Best error= ", E_best
            up_attempt = 0
            up_success = 0
        end if

    end do

    print *, "Done!"
    print *, "Best Error= ", E_best

    best_error = E_best

    w1 = full_coeffs_to_pulse(best_cos_coeffs, best_sin_coeffs, Np, tau)
    ! ! Save pulse to csv
    ! open(unit=10, file="Xpi2_20_asym_5MHz.csv", status="replace", action="write")
    ! do i = 1, Np
    !     write(10,*) w1(i)
    ! end do
    ! close(10)

    ! print *, "Best cos coeffs = ", best_cos_coeffs
    ! print *, "Best sin coeffs = ", best_sin_coeffs

    end subroutine run_annealing

end module annealing_algorithm_backend