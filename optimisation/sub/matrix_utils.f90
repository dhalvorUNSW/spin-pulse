module matrix_utils

    implicit none
    contains

    function trace2x2(mat) result(tr)

        implicit none
        complex(8), intent(in) :: mat(2,2)
        complex(8) :: tr

        ! The trace is the sum of the diagonal elements
        tr = mat(1,1) + mat(2,2)

    end function trace2x2

    function matmul2x2(A, B) result(C)
        implicit none
        complex(8), intent(in) :: A(2,2), B(2,2)
        complex(8) :: C(2,2)

        ! Perform manual 2x2 matrix multiplication
        C(1,1) = A(1,1)*B(1,1) + A(1,2)*B(2,1)
        C(1,2) = A(1,1)*B(1,2) + A(1,2)*B(2,2)
        C(2,1) = A(2,1)*B(1,1) + A(2,2)*B(2,1)
        C(2,2) = A(2,1)*B(1,2) + A(2,2)*B(2,2)
    end function matmul2x2

    function matmul2x1(A, B) result(C)
        implicit none
        complex(8), intent(in) :: A(2,2), B(2)
        complex(8) :: C(2)

        ! Perform matrix x vector multiplication
        C(1) = A(1,1)*B(1) + A(1,2)*B(2)
        C(2) = A(2,1)*B(1) + A(2,2)*B(2)

    end function matmul2x1

end module matrix_utils