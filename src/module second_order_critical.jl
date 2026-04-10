module second_order_critical

include("GeometricObjects.jl")
include("EvaluationMatrix.jl")

using LinearAlgebra
using .GeometricObjects: Riemannian_Hessian, Riemannian_grad
using .EvaluationMatrix: enumerate_quadint
using Base.Threads

export is_second_order_critical, tangent_basis

# construct a basis for the tangent space T_Q P = {H \in S^n: Tr(Q^-1 H)=0} = Q^1/2 * {matrices with zero trace}*  Q^1/2
function tangent_basis(Q::AbstractMatrix)
    n = size(Q,1)
    Qhalf = sqrt(Q)

    basis = Matrix{Float64}[]

    # --- Off-diagonal directions ---
    for i in 1:n
        for j in i+1:n
            S = zeros(n,n)
            S[i,j] = 1.0
            S[j,i] = 1.0

            push!(basis, Qhalf * S * Qhalf)
        end
    end

    # --- Diagonal trace-zero directions ---
    for i in 1:n-1
        S = zeros(n,n)
        S[i,i] = 1.0
        S[n,n] = -1.0

        push!(basis, Qhalf * S * Qhalf)
    end

    return basis
end

using LinearAlgebra

function is_second_order_critical(Q, alpha, Zmat; tol=1e-9)

    n = size(Q,1)
    Qinv = inv(Q)

    # ---- 1. Build tangent basis ----
    basis = tangent_basis(Q)   # recommended version
    m = length(basis)

    # ---- 2. Assemble Hessian matrix ----
    M = zeros(m, m)

   Threads.@threads for j in 1:m
        Hj = basis[j]

        # Hessian action
        HessHj = Riemannian_Hessian(Q, Hj, alpha, Zmat)

        for i in 1:m
            Hi = basis[i]

            # Riemannian metric:
            M[i,j] = tr(Qinv * HessHj * Qinv * Hi)
        end
    end

    # Symmetrize (numerical stability)
    M = 0.5*(M + M')

    # ---- 3. Check eigenvalues ----
    eigvals_M = eigvals(Symmetric(M))
    λmin = minimum(eigvals_M)

    return λmin ≥ -tol, λmin, eigvals_M
end

"""
Q_2 = 5^(-1/4) * [2 1 1 1
                  1 2 1 1
                  1 1 2 1
                  1 1 1 2  ]
 println("det Q_2 = ", det(Q_2))
 #inspect eigenvalues of Hessian


basis = tangent_basis(Q_2)

Zmat = enumerate_quadint(Q_2, 40)
alpha = 0.45
Qinv = inv(Q_2)
G = Riemannian_grad(Q_2, alpha, Zmat)
println("Riemannian gradient at Q_2: ")
display(G)

is_min, λmin, eigvals_M = is_second_order_critical(Q_2, alpha, Zmat)

println("Is second order critical? ", is_min)
println("Eigenvalues: ", eigvals_M)
#println("All eigenvalues: ", eigvals_M)

"""
end # module