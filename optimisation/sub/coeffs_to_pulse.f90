! Module containing functions to convert Fourier coefficients to pulse shapes

module coeffs_to_pulse
    implicit none

contains

    function cos_coeffs_to_pulse(cos_coeffs, Np, tau) result(w1)
        implicit none
        integer, intent(in) :: Np 
        real(8), intent(in) :: cos_coeffs(:)  ! Fourier coefficients
        real(8), intent(in) :: tau            ! Total duration
        real(8) :: t                          ! Time variable
        real(8) :: w
        real(8) :: w1(Np)
        real(8), parameter :: pi = 4.0d0 * atan(1.0d0)
        integer :: i, j, n_max

        w = 2.0d0 * pi / tau 
        n_max = size(cos_coeffs) - 1

        ! Construct pulse at each time
        do j = 1, Np
            t = (j - 1) * tau / dble(Np - 1)
            w1(j) = cos_coeffs(1)
            ! Compute sum over Fourier series
            do i = 1, n_max
                w1(j) = w1(j) + cos_coeffs(i + 1) * cos(dble(i) * w * t)
            end do
            w1(j) = w1(j) * w ! Scale by 1/tau (subject to change)
        end do

    end function

    function full_coeffs_to_pulse(cos_coeffs, sin_coeffs, Np, tau) result(w1)
        implicit none
        integer, intent(in) :: Np 
        real(8), intent(in) :: cos_coeffs(:), sin_coeffs(:)  ! Fourier coefficients
        real(8), intent(in) :: tau                           ! Total duration
        real(8) :: t                                         ! Time variable
        real(8) :: w
        real(8) :: w1(Np)
        real(8), parameter :: pi = 4.0d0 * atan(1.0d0)
        integer :: i, j, n_max

        w = 2.0d0 * pi / tau 
        n_max = size(cos_coeffs) - 1

        ! Construct pulse at each time
        do j = 1, Np
            t = (j - 1) * tau / dble(Np - 1)
            w1(j) = cos_coeffs(1)
            ! Compute sum over Fourier series
            do i = 1, n_max
                w1(j) = w1(j) + cos_coeffs(i + 1) * cos(dble(i) * w * t) + sin_coeffs(i) * sin(dble(i) * w * t)
            end do
            w1(j) = w1(j) * w ! Scale by 1/tau (subject to change)
        end do

    end function

end module coeffs_to_pulse