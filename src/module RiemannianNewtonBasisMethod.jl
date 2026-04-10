module RiemannianNewtonBasisMethod
using LinearAlgebra
using JLD2
using Base.Threads
using LoopVectorization
#using Revise

include("GeometricObjects.jl")
using .GeometricObjects: Riemannian_Hessian, Riemannian_grad,compute_E_energy, exponentialRetraction

export solve_newton_eq_basis

# want to solve the newton equation H_Q[ξ] = - G_Q for ξ ∈ T_Q P_n
# using linearlity of Hessian and write ξ = sum_{i=1}^k α_i ξ_i for some basis {ξ_i} of T_Q P_n
# then we have to solve the linear system \sum_{i=1}^k α_i H_Q[ξ_i] = - G_Q 
"""

    norm_Q(A, Q, metric_type)

Compute the squared norm of tangent vector A at point Q on the manifold:

- affine-invariant metric: ⟨A,A⟩_Q = Tr(Q⁻¹ A Q⁻¹ A)
- Euclidean metric: ⟨A,A⟩ = Tr(A*A)

Arguments:
- A           : n×n tangent vector (symmetric)
- Q           : n×n SPD matrix
- metric_type : 1 = affine-invariant, 0 = Euclidean

Returns:
- val : scalar
"""
function norm_Q(A::AbstractMatrix{<:Real}, Q::AbstractMatrix{<:Real}, metric_type::Int)
    if metric_type == 1
        # Solve Q * X = A → X = Q⁻¹ A
        Qinv_A = Q \ A
        val = tr(Qinv_A * Qinv_A)
    else
        val = tr(A * A)
        if val < 0
            val = abs(val)
        end
    end
    return val
end

"""
Project a symmetric matrix X onto the tangent space at Q
for the SPD manifold with det(Q) = 1.

T_Q = { H symmetric | tr(Q^{-1} H) = 0 }
"""
function tangentProjection(X::AbstractMatrix, Q::AbstractMatrix)
    n = size(Q, 1)

    # Cholesky factor Q = R'R
    R = cholesky(Q).U

    # Compute Q^{-1} X using Cholesky solves
    QiX = R \ (R' \ X)

    # Remove trace component
    Y = X - (tr(QiX) / n) * Q

    return Y
end

"""
function that vectorizes a symmetric matrix by staking the upper triangular part into a vector

"""
function vech_sym(X::AbstractMatrix)
    n = size(X, 1)
    vech = Vector{Float64}(undef, div(n*(n+1), 2))
    idx = 1
    for j in 1:n
        for i in 1:j
            vech[idx] = X[i, j]
            idx += 1
        end
    end
    return vech
end


 """
 tangent_basis(Q) returns a basis of the tangent space T_Q P_n at Q
 """    
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

"""
Calculate the Hessian of the basis elements

"""
function hessian_basis_elements(Q::AbstractMatrix, basis::Vector{<:AbstractMatrix}, alpha, Zmat)
    hessian_basis = Vector{Matrix{Float64}}(undef, length(basis))
    for i in 1:length(basis)
        H = Riemannian_Hessian(Q, basis[i], alpha, Zmat)
        hessian_basis[i] = H
    end
    return hessian_basis
end

#basis = tangent_basis(Qs[1])

#hessian_basis = hessian_basis_elements(Qs[1], basis, alpha, Zmat)

"""
function that transforms Newton Equation into a linear system by vectorizing the Hessian and gradient
 and using the basis of the tangent space to write the search direction as a linear combination of the basis elements
"""
function newton_equation_to_linear_system(Q::AbstractMatrix, G_Q::AbstractMatrix,Hessian_basis::Vector{<:AbstractMatrix})
         n = size(Q, 1)
         k = length(Hessian_basis) # = n(n+1)/2 - 1
         # Vectorize the gradient
         g = vech_sym(G_Q)

         # Initialize the Hessian matrix for the linear system
         H = zeros(k + 1, k)
         # Fill the Hessian matrix by applying the Hessian to each basis element and vectorizing the result
         for i in 1:k
             H[:, i] = vech_sym(Hessian_basis[i])
         end
   return g, H
end

"""
function that solves the linear system H * α = -g for the coefficients α of the search direction in the basis
    Input: Q - current point
        alpha - E-energy parameter
        Zmat - Partial Evaluation Matrix
        
    Output: tangent vector ξ = ∑_{i=1}^k α_i ξ_i in T_Q P_n that solves 
            the Newton equation H_Q[ξ] = - G_Q        
"""

function solve_newton_eq_basis(Q::AbstractMatrix,alpha::Real, Zmat::AbstractMatrix)
    G_Q = Riemannian_grad(Q, alpha, Zmat)
    basis = tangent_basis(Q)
    Hessian_basis = hessian_basis_elements(Q, basis, alpha, Zmat)
    g, H = newton_equation_to_linear_system(Q, G_Q, Hessian_basis)
    # slve the equation H * α = -g for the coefficients α, a solution exists
    # rank(H) = k, since the Hessian is positive definite, so we can use the pseudo-inverse to solve for α
     if rank(H) < size(H, 2)
        #@warn "Hessian matrix is not full rank, using pseudo-inverse to solve for α"
        α = -pinv(H) * g
    else
        α = -H \ g
    end
    # construct the search direction ξ = ∑_{i=1}^k α_i ξ_i
    ξ = zeros(size(Q))
    for i in 1:length(basis)
        ξ += α[i] * basis[i]
    end
    ξ = tangentProjection(ξ, Q) # ensure ξ is in T_Q P_n by projecting onto the tangent space
    return ξ
   
end

function solve_newton_eq_basis_least_sqares(Q::AbstractMatrix,alpha::Real, Zmat::AbstractMatrix)
    G_Q = Riemannian_grad(Q, alpha, Zmat)
    basis = tangent_basis(Q)
    Hessian_basis = hessian_basis_elements(Q, basis, alpha, Zmat)
    g, H = newton_equation_to_linear_system(Q, G_Q, Hessian_basis)
    # slve the equation H * α = -g for the coefficients α, a solution exists
    # rank(H) = k, since the Hessian is positive definite, so we can use the pseudo-inverse to solve for α
     if rank(H) < size(H, 2)
        #@warn "Hessian matrix is not full rank, using pseudo-inverse to solve for α"
        α = -pinv(H) * g
    else
        U, S, V = svd(H)
        α = -V * Diagonal(1 ./ S) * U' * g
    end
    # construct the search direction ξ = ∑_{i=1}^k α_i ξ_i
    ξ = zeros(size(Q))
    for i in 1:length(basis)
        ξ += α[i] * basis[i]
    end
    ξ = tangentProjection(ξ, Q) # ensure ξ is in T_Q P_n by projecting onto the tangent space
    return ξ
   
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

""" Stopping criterion for stagnation of the gradient norm
    Checks if the gradient norm has not decreased for the last i iterations
    calculate the average of the last i gradient norms and compare it to the current gradient norm.
"""
function norm_grad_stagnation(grad_norm_arr, iter, i=10, tol=1e-8)
    if iter < i + 1    
        return false
    end
    current_grad = grad_norm_arr[iter]
    i_grad = grad_norm_arr[iter-i]
    #println("difference in gradient norm: ", abs(current_grad - i_grad))
    return abs(current_grad - i_grad) < tol
end


"""
    metric_Q(A, B, Q, metric_type)

Compute the Riemannian metric between tangent vectors A and B at point Q:

- affine-invariant metric: ⟨A,B⟩_Q = Tr(Q⁻¹ A Q⁻¹ B)
- Euclidean metric: ⟨A,B⟩ = Tr(A*B)

Arguments:
- A, B        : n×n tangent vectors
- Q           : n×n SPD matrix
- metric_type : 1 = affine-invariant, 0 = Euclidean

Returns:
- val : scalar
"""
function metric_Q(A::AbstractMatrix{<:Real},
                  B::AbstractMatrix{<:Real},
                  Q::AbstractMatrix{<:Real},
                  metric_type::Int)
    if metric_type == 1
        Qinv_A = Q \ A
        Qinv_B = Q \ B
        val = tr(Qinv_A * Qinv_B)
    else
        val = tr(A * B)
    end
    return val
end

"""
run accelerated line search along the geodesic defined by the search direction ξ 
to find a new point Q_new = Retr(Q, ξ) with lower energy

metric_type = 1 # affine-invariant metric
evaluationUpdate = false #no internal update of PEM
opts = Dict(:alpha_bar => 1.0, :c => 1e-4, :beta => 0.5, :maxIter => 100, :tol => 5e-15, :C => C)
E = (Q, Z) -> compute_E_energy(Q, alpha, Z)
grad_E = (Q, Z) -> Riemannian_grad(Q, alpha, Z)
Retr = (Q, H) -> exponentialRetraction(Q, H)
"""

function Riemannian_Newton_Method(Q0, E, grad_E, Retr, opts, metric_type, Zmat, evaluationUpdate::Bool=false)
    # ------------------ Extract options ------------------
    alpha_bar = opts[:alpha_bar]
    c         = opts[:c]
    beta      = opts[:beta]
    maxIter   = opts[:maxIter]
    tol       = opts[:tol]
    alpha     = opts[:alpha]
    # ------------------ Initialization ------------------
    tol_truncation = 1e-100
    Qk = Q0
    Zmat_original = copy(Zmat)
    Zmat =  truncate_Evaluation_Matrix_gradient(Qk, Zmat_original, opts[:alpha], tol_truncation) #1e-100
    n = size(Qk, 1)
    iter = 1
    grad_norm_arr = zeros(Float64, maxIter+1)
    Gk =  grad_E(Qk, Zmat)
    grad_norm_arr[iter] = sqrt(abs(norm_Q(Gk, Qk, metric_type)))

    E_Q_arr       = zeros(Float64, maxIter+1)
    E_Q_arr[iter] =  E(Qk, Zmat)
    evaluationSize_1 = size(Zmat, 1)
    evaluationSize_2 = 0
    # ------------------ Main loop ------------------
    
   for k in 1:maxIter
        # truncate universal evaluation matrix for values e^-α Q[z] zz^T< 1e-50
        if k % 4 == 0 && k != 1 && n>=6
        Zmat =  truncate_Evaluation_Matrix_gradient(Qk, Zmat_original, opts[:alpha], tol_truncation) #1e-100
        end
  
        # ------------------ Stopping condition ------------------
        if   grad_norm_arr[k] <= tol
            println("*** Converged at iteration $k| E_Q = $(round(E_Q_arr[k], sigdigits=4))| norm_grad = $(round(grad_norm_arr[k], sigdigits=4)) | Evaluation size $(size(Zmat, 1))")
            break 
        end

        if iter >=2 && evaluationUpdate
            evaluationSize_1 = evaluationSize_2
        end
        
        # Determine search direction solving the Newton Equation
        eta_k =  solve_newton_eq_basis_least_sqares(Qk, alpha, Zmat)
        # Metric-related quantity for Armijo
        #Gk = grad_E(Qk, Zmat)
        m = metric_Q(Gk, eta_k, Qk, 1) #<G_k, eta_k>_Qk
        m = min(m, -1e-14)
        t = min(alpha_bar, 1 / grad_norm_arr[k])
        E_Qk = E_Q_arr[k]
 
        # ------------------ Backtracking line search ------------------
        Q_trial = copy(Qk)
        E_trial = copy(E_Qk)
        while true
            Q_trial = Retr(Qk, t * eta_k)
            E_trial = E(Q_trial, Zmat)
            #println("det Q_trial : ", det(Q_trial))
            # Armijo condition
           if E_trial <= (E_Qk + c * t * m) && isposdef(Q_trial) && abs(det(Q_trial)-1) < 1e-5
            #println("final step size t: ", t)
               break
           else
               t *= beta
               if t < 1e-16  # stepsize too small 1e-50
                   break
               end
           end
       end

        # Update iterate if Q_trial is positive definite, otherwise keep Qk and break to avoid numerical issues
        if isposdef(Q_trial)
            Qk = Q_trial
            E_Q_arr[k+1] = E_trial
            Gk = grad_E(Qk, Zmat)
            grad_norm_arr[k+1] = sqrt(abs(norm_Q(Gk, Qk, metric_type)))
        else
            println(":-( Warning: Obtained non-positive definite matrix at iteration $k, stopping optimization.")
            iter = maxIter # to indicate no convergence
            break
        end
        
        # stopping criterion for non-increasing gradient norm
        if norm_grad_stagnation(grad_norm_arr, k, 5, 1e-10) || E_Q_arr[k] < 1e-3
             println(":-( No decrease in energy at iter $k | E_Q=$(round(E_Q_arr[k], sigdigits=4)) | grad norm=$(round(grad_norm_arr[k], sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors " )
             iter = maxIter # to indicate no convergence
             break
        end


        if evaluationUpdate
          Zmat = enumerate_quadint(Qk, opts[:C]) # update Zmat if needed
          evaluationSize_2 = size(Zmat, 1)
        end

        if iter >= 2 && evaluationUpdate
            if evaluationSize_1 == evaluationSize_2
                evaluationUpdate = false # stop updating if size didn't change
            end
        end

        # ------------------ Progress report ------------------
        if k == 1|| k % 10 == 0
           println("Iter $k | E_Q = $(round(E_Qk, sigdigits=4))| grad norm = $(round(grad_norm_arr[k], sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors")
        end
        iter = iter + 1
    end
    
    return Qk, E_Q_arr[1:(iter)], grad_norm_arr[1:(iter)], iter
end

end #module RiemannianNewtonBasisMethod