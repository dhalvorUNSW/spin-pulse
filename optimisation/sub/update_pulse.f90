! Module contatining functions to update an arbitray pulse shape

module update_pulse
    implicit none
    integer :: nextIndex = 1  ! Initialize nextIndex as a module variable

contains

    function update_coeffs(coeffs, step_sizes) result(new_coeffs)
        implicit none
        real(8), intent(in) :: coeffs(:)  ! Fourier coefficients (modified in place)
        real(8), intent(in) :: step_sizes(:)    ! Step sizes for updating coefficients
        integer :: n_max
        real(8) :: new_coeffs(size(coeffs))        ! Output new Fourier coefficients
        real(8) :: R1                           ! Random number for update

        n_max = size(coeffs)
        new_coeffs = coeffs

        call random_number(R1)
        R1 = 2.0d0 * R1 - 1.0d0  ! Scale the random number to [-1, 1]

        ! Update Fourier coeffs
        new_coeffs(nextIndex) = new_coeffs(nextIndex) + R1 * step_sizes(nextIndex)

        ! Update next index
        nextIndex = nextIndex + 1
        if (nextIndex > n_max) then
            nextIndex = 1
        end if

    end function

end module update_pulse