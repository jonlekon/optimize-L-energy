
module OptimalLattices
using LinearAlgebra
export basisGenerator, optimal_gram_matrices
# n = 2
# Hexagonal lattice
function B_hexagonal_generator()
    B = [
        1.0  0.5;
        0.0  sqrt(3)/2
    ]
# normalize determinant
    B /= det(B)^(1/2)
    
    
end
# n = 3
# Face-centered cubic (FCC) lattice
function B_FCC_generator()
    Q = [
        2 0 1
        0 2 -1
        1 -1 2
    ]
# normalize determinant
    Q /= 2^(2/3)
    return cholesky(Q).U
end
#Q_FCC = B_FCC_generator()' * B_FCC_generator()
#println("det(Q_FCC) = ", det(Q_FCC))

# n= 4
# D4 optimal lattice generator matrix
function D4_generator()
    B_D4 = [
        1  0  0  0;
       -1  1  0  0;
        0  -1  1  1;
        0  0  -1 1 
    ]
    B_D4 /= det(B_D4)^(1/4)
    return B_D4
end
#Q_D4 = D4_generator()' * D4_generator()
#println("det(Q_D4) = ", det(Q_D4))

# n = 5
# A5^2 lattice 
function A5up2_generator()
    Q =[3 -1 1 1 -1;
       -1 3 -1 1 1;
        1 -1 3 -1 1;
        1 1 -1 3 -1;
        -1 1 1 -1 3]
    Q /= det(Q)^(1/5)
    B = cholesky(Q).U
    return B
end
#Q_D5 = D5_generator()' * D5_generator()
#println("det(Q_D5) = ", det(Q_D5))
# n= 6
# D6 optimal lattice generator matrix
function D6_generator()
    B_D6 = [
        1  1  0  0  0  0;
        1 -1  0  0  0  0;
        0  0  1  1  0  0;
        0  0  1 -1  0  0;
        0  0  0  0  1  1;
        0  0  0  0  1 -1
    ]
    B_D6 /= abs(det(B_D6))^(1/6)
    return B_D6
end
#Q_D6 = D6_generator()' * D6_generator()
#println("det(Q_D6) = ", det(Q_D6))
# n= 7
# D7 optimal lattice generator matrix
function E7_generator() 
    #not root lattice A7, A7*, D7, D7*
    # I_7, E_7 is optimal
 Q = [
    2 -1 0 0 0 0 0;
    -1 2 -1 0 0 0 0;
    0 -1 2 -1 0 0 0;
    0 0 -1 2 -1 0 -1;
    0 0 0 -1 2 -1 0;
    0 0 0 0 -1 2 0;
    0 0 0 -1 0 0 2;
 ]
    B_D7 =cholesky(Q).U

    B_D7 /= abs(det(B_D7))^(1/7)
    return B_D7
end
#Q_D7 = D7_generator()' * D7_generator()
#println("det(Q_D7) = ", det(Q_D7))
# n= 8

# E8 optimal lattice generator matrix
function E8_generator()

    G = [
        2   -1    0    0    0    0    0   1/2;
        0    1   -1    0    0    0    0   1/2;
        0    0    1   -1    0    0    0   1/2;
        0    0    0    1   -1    0    0   1/2;
        0    0    0    0    1   -1    0   1/2;
        0    0    0    0    0    1   -1   1/2;
        0    0    0    0    0    0    1   1/2;
        0    0    0    0    0    0    0   1/2
    ]

end

# function that picks optimal lattice basis depending on 
function basisGenerator(n::Int)
    if n == 2
        B = B_hexagonal_generator()
    elseif n == 3
        B = B_FCC_generator()
    elseif n == 4
        B = D4_generator()
    elseif n == 5
        B = A5up2_generator()
    elseif n == 7
        B = E7_generator()
    elseif n == 6
        B = D6_generator()
    elseif n == 8
        B = E8_generator()
    else
        error("No optimal lattice generator defined for n = $n")
    end
    return B
end

### optimal lattices obtained by smooth optimization

function optimal_gram_matrices(n::Int)
    symmetrize(A) = A + A' - Diagonal(diag(A))  # fill lower triangle

    if n == 2
        Q = (1 / sqrt(3)) * symmetrize([
            4  1;
            0  4
        ])
        return [Q]

    elseif n == 3
        Q1 = (1 / 2^(2/3)) * symmetrize([
            2  -1  -1;
            0   2   1;
            0   0   2
        ])

        Q2 = (1 / 2^(4/3)) * symmetrize([
            3   1   1;
            0   3  -1;
            0   0   3
        ])

        return [Q1, Q2]

    elseif n == 4
        Q1 = (1 / 2^(1/2)) * symmetrize([
            2  1  1  -1;
            0  2  0   0;
            0  0  2   0;
            0  0  0   2
        ])

        Q2 = (1 / 5^(1/4)) * symmetrize([
            2  1  1  1;
            0  2  1  1;
            0  0  2  1;
            0  0  0  2
        ])

        return [Q1, Q2]

    elseif n == 5
        Q1 = (1 / 2^(8/5)) * symmetrize([
            4  -2  -2  -2   2;
            0   5  -8  -8   3;
            0   0   5   8   8;
            0   0   0   5   8;
            0   0   0   0   5
        ])

        Q2 = (1 / 2^(2/5)) * symmetrize([
            2  -1   1  -1   1;
            0   2   0   1   0;
            0   0   2  -1   1;
            0   0   0   2   0;
            0   0   0   0   2
        ])

        return [Q1, Q2]

    elseif n == 6
        Q1 = (1 / 3^(5/6)) * symmetrize([
            4   2  -2   1   1   1;
            0   4  -1   2  -1   2;
            0   0   4  -2   1  -2;
            0   0   0   4  -2   1;
            0   0   0   0   4  -2;
            0   0   0   0   0   4
        ])

        Q2 = (1 / 3^(1/6)) * symmetrize([
            2  -1  -1  -1   1   1;
            0   2   1   0   0   0;
            0   0   2   1   0  -1;
            0   0   0   2  -1  -1;
            0   0   0   0   2   1;
            0   0   0   0   0   2
        ])

        return [Q1, Q2]

    elseif n == 7
        Q1 = (1 / 2^(6/7)) * symmetrize([
            3  -1  -1  -1   1  -1   1;
            0   3  -1   1  -1   1   1;
            0   0   3  -1  -1   1  -1;
            0   0   0   3   1  -1   1;
            0   0   0   0   3  -1   1;
            0   0   0   0   0   3  -1;
            0   0   0   0   0   0   3
        ])

        Q2 = (1 / 2^(1/7)) * symmetrize([
            2  -1   1   1  -1  -1   1;
            0   2   0   0   1   1  -1;
            0   0   2   1  -1  -1   1;
            0   0   0   2  -1   0   0;
            0   0   0   0   2   1  -1;
            0   0   0   0   0   2  -1;
            0   0   0   0   0   0   2
        ])

        return [Q1, Q2]

    elseif n == 8
        Q1 = (1 / 2^(1/7)) * symmetrize([
            2  -1   1  -1  -1   1   1  -1;
            0   2  -1   0   0  -1   0   0;
            0   0   2  -1   0   1   1  -1;
            0   0   0   2   1  -1  -1   1;
            0   0   0   0   2  -1  -1   1;
            0   0   0   0   0   2   1  -1;
            0   0   0   0   0   0   2  -1;
            0   0   0   0   0   0   0   2
        ])

        return [Q1]

    else
        error("No optimal Gram matrices stored for n = $n")
    end
end

end # module OptimalLattices