#Perturb an isolated minimum along the direction of the eigenvector of its Hessian matrix
# Qpert = Q + epsilon * ξ
# ξ is in T_Q P such that < Hess E(Q)[ξ], ξ >_Q = 0


src = joinpath(@__DIR__, "..", "src")
include(joinpath(src, "GeometricObjects.jl"))
include(joinpath(src, "EvaluationMatrix.jl"))
include(joinpath(src, "module second_order_critical.jl"))

using LinearAlgebra
using .GeometricObjects: Riemannian_Hessian, exponentialRetraction, Riemannian_grad, compute_E_energy
using .EvaluationMatrix: enumerate_quadint
using .second_order_critical: is_second_order_critical


alpha = 0.437
C = 200
Q = (1 / 2^(2/3)) * [
            2   -1   -1;
           -1    2    1;
           -1    1    2
        ]
Zmat = enumerate_quadint(Q, C)
#---------------functions-------------------
#built std basis {ξ_1,...,ξ_{d-1}} of T_Q P_n
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
basis = tangent_basis(Q)
#display(basis)

#build the Hessian basis {H[ξ_1],...,H[ξ_{d-1}]} of T_Q P_n
function hessian_basis_elements(Q::AbstractMatrix, basis::Vector{<:AbstractMatrix}, alpha, Zmat)
    hessian_basis = Vector{Matrix{Float64}}(undef, length(basis))
    for i in 1:length(basis)
        H = Riemannian_Hessian(Q, basis[i], alpha, Zmat)
        hessian_basis[i] = H
    end
    return hessian_basis
end
Hbasis = hessian_basis_elements(Q, basis, alpha, Zmat)
#display(Hbasis)


#Inner product <A,B>_Q
function inner_Q(A, B, Q; use_Q_metric=false)
    if use_Q_metric
        Qi = inv(Q)
        return tr(Qi * A * Qi * B)
    else
        return tr(A * B)  # Frobenius
    end
end


#Build matrix M_{ij} = <H[ξ_i], ξ_j>_Q
function build_M(Q, basis, Hbasis; use_Q_metric=false)
    m = length(basis)
    M = zeros(Float64, m, m)

    for i in 1:m
        for j in i:m
            M[i,j] = inner_Q(Hbasis[i], basis[j], Q; use_Q_metric=use_Q_metric)
            M[j,i] = M[i,j]  # symmetry
        end
    end

    return Symmetric(M)
end
 M = build_M(Q, basis, Hbasis; use_Q_metric=false)
#display(M)


#Find x such that x'Mx ≈ 0
function find_null_directions(M; tol=1e-8)
    F = eigen(M)
    λ = F.values
    V = F.vectors

    idx = findall(abs.(λ) .< tol)

    return λ[idx], V[:, idx]
end

# solve for x such that x'Mx ≈ 0
function compute_flat_x(Q, basis, Hbasis; tol=1e-8, use_Q_metric=false)
    M = build_M(Q, basis, Hbasis; use_Q_metric=use_Q_metric)

    λ, vecs = find_null_directions(M; tol=tol)

    println("Near-zero eigenvalues:")
    println(λ)

    return vecs  # each column is a solution x
end
x = compute_flat_x(Q, basis, Hbasis; tol=1e-8, use_Q_metric=false)
#display(flat_vecs)

# build ξ from coefficients x
function build_xi(x, basis)
    ξ = zero(basis[1])
    for i in eachindex(basis)
        ξ += x[i] * basis[i]
    end
    return ξ
end
 ξ = build_xi(x[:,1], basis)
 println("tr(inv(Q)*ξ): ", tr(inv(Q) * ξ)) # check trace zero

# apply perturbation
function perturb_Q(Q, x, basis, ε)
    ξ = build_xi(x, basis)
    return exponentialRetraction(Q, ε * ξ)
end

epsilon = 0.01 #0.0005 is best
Qpert = perturb_Q(Q, x[:,1], basis, epsilon)
println("Qpert = Q + ε * ξ for ε = ", epsilon)
#println("det(Q_pert): ", det(Qpert))
#println("Q_pert is positive definite: ", isposdef(Qpert))

#check energy and gradient at Q_pert
Zpert = enumerate_quadint(Qpert, C)
E_pert = compute_E_energy(Qpert, alpha, Zpert)
G_pert = Riemannian_grad(Qpert, alpha, Zpert)
norm_G_pert = sqrt(tr((Qpert\G_pert)*(Qpert\G_pert)'))
println("E-energy at Q_pert: ", E_pert)
println("Norm of Riemannian gradient at Q_pert: ", norm_G_pert)

#check second order criticality at Q_pert
_, _, lambdas_pert = is_second_order_critical(Qpert, alpha, Zpert)
println("Eigenvalues of the Hessian at Q_pert: ", sort(round.(lambdas_pert; sigdigits=4)))

Qfact = 2^(2/3) * Qpert
println("Qpert factored by 2^(2/3) to check if it is close to the original Q: ")
display(Qfact)




