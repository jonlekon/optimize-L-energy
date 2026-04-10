#works
src = joinpath(@__DIR__, "..", "src")
input_data = joinpath(@__DIR__, "..", "input_data")

include(joinpath(src, "GeometricObjects.jl"))
include(joinpath(src, "EvaluationMatrix.jl"))
include(joinpath(src, "module second_order_critical.jl"))

using .GeometricObjects
using JLD2
using DelimitedFiles
using LinearAlgebra
using Base.Threads
using JLD2
using LoopVectorization
using Statistics
using Revise
using .EvaluationMatrix: enumerate_quadint
using .second_order_critical: is_second_order_critical

function optimal_gram_matrices(n::Int)
    symmetrize(A) = A + A' - Diagonal(diag(A))  # fill lower triangle

    if n == 2
        Q =  [
            2  -1;
            -1  2
        ]
        Q = det(Q)^(-1/2) *Q
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
        Q2 = (1 / 2)^(8/5) * [    
             4    -2     0     0     0;
            -2     5    -2     2     2;
            0    -2     4     0     0;
            0     2     0     4     0;
            0     2     0     0     4]

        Q1 = (1/2)^(2/5)* [
            2  -1  1  1  -1;
            -1  2  -1  0  0;
            1  -1  2  0  -1;
            1  0  0  2  -1;
            -1  0  -1  -1  2];

        return [Q1, Q2]

    elseif n == 6
        Q2 = (1 / 3^(5/6)) * symmetrize([
            4   2  -2   1   1   1;
            0   4  -1   2  -1   2;
            0   0   4  -2   1  -2;
            0   0   0   4  -2   1;
            0   0   0   0   4  -2;
            0   0   0   0   0   4
        ])

        Q1 = (1/3)^(1/6)* [2  -1  -1  -1  1  1;
                    -1  2  1  0  0  0;
                    -1  1  2  1  0  -1;
                    -1  0  1  2  -1  -1;
                    1  0  0  -1  2  1;
                    1  0  -1  -1  1  2]

        return [Q1, Q2]

    elseif n == 7
        Q2 = (1 / 2^(6/7)) * symmetrize([
            3  -1  -1  -1   1  -1   1;
            0   3  -1   1  -1   1   1;
            0   0   3  -1  -1   1  -1;
            0   0   0   3   1  -1   1;
            0   0   0   0   3  -1   1;
            0   0   0   0   0   3  -1;
            0   0   0   0   0   0   3
        ])

        Q1 = (1 / 2^(1/7)) * symmetrize([
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
        Q1 =  symmetrize([
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
function radius_C(n)
    if n==2
        C = 100
    elseif n==3
        C = 100
    elseif n==4
        C = 70
    elseif n==5
        C = 30
    elseif n==6
        C= 20
    elseif n==7
        C = 14
    elseif n==8
        C = 12
    else
            error("No C nor n=", n)
    end
 return C

end


function truncate_Evaluation_Matrix_gradient(Q, Zmat, α, threshold)

    ZQ = Zmat * Q
    ZQ_norm2 = sum(ZQ.^2, dims=2)

    U = cholesky(Q).U
    ZU = Zmat * U

    n, m = size(ZU)
    mask = Vector{Bool}(undef, n)

    @threads for i in 1:n
        s = 0.0

        @turbo for j in 1:m
            v = ZU[i,j]
            s += v*v
        end

        w = exp(-α*s)

        contrib = α * w * ZQ_norm2[i]

        mask[i] = contrib >= threshold
    end

    return Zmat[mask,:]
end



n = 4
C = 70
alpha = float(pi) # parameter alpha in the energy functional
println("Using C = $C")

#------------------------Load universal PEM-------------------------------------------------
#folder_PEM = "Partial_Evaluation_Lattices/"
#@load joinpath(folder_PEM, "universal_PEL_n$(n)_C$(C).jld2") PEM
#Z = PEM
#println("Size of PEL is : ",size(Z,1))

Qs = optimal_gram_matrices(n)
N = length(Qs)

# output E_alpha(Q), || grad E_alpha(Q) ||_Q and Hessian EVs for each optimal Q
for i = 1:N
  Q = Qs[i]
  Zmat =enumerate_quadint(Q, C) #truncate_Evaluation_Matrix_gradient(Q, Z, alpha, threshold)
  println("Size of PEL is : ",size(Zmat,1))

  E = compute_E_energy(Q, alpha, Zmat)
  G = Riemannian_grad(Q, alpha, Zmat)
  norm_G = sqrt(tr((Q\G)*(Q\G)'))
  _,_,lambdas = is_second_order_critical(Q, alpha, Zmat)

  println("E-energy at Q_$(n)^{($(i))} is: ", E)
  println("Norm Gradient at Q_$(n)^{($(i))} is: ", norm_G)
  println("Eigenvalues of the Hessian matrix at Q_$(n)^($(i)) are: ", sort(round.(lambdas; sigdigits=4)))
end

"""
n = 2:
Q_2^(1) = 2.6666 0.666;
          0.666 2.666 -> hexgonal lattice

n= 3:
Q_3^(1)=
 0.793701  -0.39685   -0.39685
 -0.39685    0.793701   0.39685
 -0.39685    0.39685    0.793701
det Q_renorm: 0.25

Q_3^(2)=
 0.75   0.25   0.25
 0.25   0.75  -0.25
 0.25  -0.25   0.75
det Q_renorm: 0.25

n = 4
Q_4^(1)=
   0.707107  0.353553  0.353553  -0.353553
  0.353553  0.707107  0.0        0.0
  0.353553  0.0       0.707107   0.0
 -0.353553  0.0       0.0        0.707107
det Q_renorm: 0.0625

Q_4^(2)=
0.66874  0.33437  0.33437  0.33437
 0.33437  0.66874  0.33437  0.33437
 0.33437  0.33437  0.66874  0.33437
 0.33437  0.33437  0.33437  0.66874
det Q_renorm: 0.0625

n = 5
Q_5^(1)=
  0.608364  -0.304182  -0.304182  -0.304182  0.304182
 -0.304182   0.760455  -1.21673   -1.21673   0.456273
 -0.304182  -1.21673    0.760455   1.21673   1.21673
 -0.304182  -1.21673    1.21673    0.760455  1.21673
  0.304182   0.456273   1.21673    1.21673   0.760455
det Q_renorm: 2.53125

Q_5^(2)=
 0.698827  -0.349414   0.349414  -0.349414  0.349414
 -0.349414   0.698827   0.0        0.349414  0.0
  0.349414   0.0        0.698827  -0.349414  0.349414
 -0.349414   0.349414  -0.349414   0.698827  0.0
  0.349414   0.0        0.349414   0.0       0.698827
det Q_renorm: 0.020833333333333346

n = 6
Q_6^(1)=
  0.800625   0.400312  -0.400312   0.200156   0.200156   0.200156
  0.400312   0.800625  -0.200156   0.400312  -0.200156   0.400312
 -0.400312  -0.200156   0.800625  -0.400312   0.200156  -0.400312
  0.200156   0.400312  -0.400312   0.800625  -0.400312   0.200156
  0.200156  -0.200156   0.200156  -0.400312   0.800625  -0.400312
  0.200156   0.400312  -0.400312   0.200156  -0.400312   0.800625
det Q_renorm: 0.015625

Q_6^(2)=
  0.832683  -0.416342  -0.416342  -0.416342   0.416342   0.416342
 -0.416342   0.832683   0.416342   0.0        0.0        0.0
 -0.416342   0.416342   0.832683   0.416342   0.0       -0.416342
 -0.416342   0.0        0.416342   0.832683  -0.416342  -0.416342
  0.416342   0.0        0.0       -0.416342   0.832683   0.416342
  0.416342   0.0       -0.416342  -0.416342   0.416342   0.832683
det Q_renorm: 0.015625

n = 7
Q_7^(1), Q_7^(2) are the same

n = 8
Q_5^(1) is the same -> E_8 

"""


