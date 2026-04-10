module RiemannianGradientDescent

using LinearAlgebra
using Base.Threads
using LinearAlgebra
using JLD2
using DelimitedFiles
#using Revise
using LoopVectorization

include("EvaluationMatrix.jl")
include("GeometricObjects.jl")
using .EvaluationMatrix: enumerate_quadint
using .GeometricObjects: compute_E_energy, Riemannian_grad, exponentialRetraction

export AcceleratedLineSearch_P



function truncate_Evaluation_Matrix_gradient(Q, Zmat, α, threshold)

    ZQ = Zmat * Q
    ZQ_norm2 = sum(ZQ.^2, dims=2)

    U = cholesky(Q).U
    ZU = Zmat * U

    n, m = size(ZU)
    mask = Vector{Bool}(undef, n)

    for i in 1:n # @threads
        s = 0.0

         for j in 1:m # @turbo
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

    AcceleratedLineSearch_P(Q0, E, grad_E, Retr, opts, metric_type, Zmat)

Accelerated Line Search (ALS) on a Riemannian manifold P.

Arguments:
- Q0          : initial point on the manifold
- E           : function E(Q, Zmat) returning scalar energy
- grad_E      : function grad_E(Q, Zmat) returning Riemannian gradient
- Retr        : retraction function Retr(Q, eta)
- opts        : struct or NamedTuple with fields:
                  alpha_bar  : maximum step size (>0)
                  c          : Armijo constant in (0,1)
                  beta       : contraction ratio (0,1)
                  maxIter    : maximum iterations
                  tol        : convergence tolerance
- metric_type : metric type identifier for `norm_Q` / `metric_Q`
- Zmat        : truncated lattice, passed to energy/gradient functions

Returns:
- Qk          : final point
- E_Q_arr     : array of energy values
- grad_norm_arr: array of gradient norms
- k           : number of iterations performed
"""
function AcceleratedLineSearch_P(Q0, E, grad_E, Retr, opts, metric_type, Zmat_original, evaluationUpdate::Bool=false)
    # ------------------ Extract options ------------------
    alpha_bar = opts[:alpha_bar]
    c         = opts[:c]
    beta      = opts[:beta]
    maxIter   = opts[:maxIter]
    tol       = opts[:tol]
    alpha     = opts[:alpha]
    # ------------------ Initialization ------------------
    tol_truncation = 1e-60
    Qk = Q0
    Zmat =  truncate_Evaluation_Matrix_gradient(Qk, Zmat_original, alpha, tol_truncation)
    n = size(Qk, 1)
    iter = 1
    grad_norm_arr = zeros(Float64, maxIter + 1)
    Gk = grad_E(Qk, Zmat)
    grad_norm_arr[iter] = sqrt(abs(norm_Q(Gk, Qk, metric_type)))
    E_Q_arr       = zeros(Float64, maxIter + 1)
    E_Q_arr[iter] = E(Qk, Zmat)
    evaluationSize_1 = size(Zmat, 1)
    evaluationSize_2 = 0
    #println("E_Q_0: ",E_Q_arr[iter] )
    #println("norm grad_0 : ",grad_norm_arr[iter])
    # ------------------ Main loop ------------------
    
   for k in 1:maxIter
        # truncate universal evaluation matrix for values e^-α Q[z] zz^T< 1e-50
         if k % 25 == 0 && k != 1 && n>= 6
        Zmat =  truncate_Evaluation_Matrix_gradient(Qk, Zmat_original, alpha, tol_truncation)
        end
         # ------------------ Stopping condition ------------------
        if  grad_norm_arr[k] <= tol
            println("**** Converged at iteration $k| E_Q = $(round(E_Q_arr[k], sigdigits=4))| grad norm = $(round(grad_norm_arr[k], sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors")
            break
        end

        if iter >=2 && evaluationUpdate
            evaluationSize_1 = evaluationSize_2
        end
        
        # Determine search direction (normalized)
        Gk = tangentProjection(Gk, Qk)
        eta_k = -Gk  / max(grad_norm_arr[k] , eps())

        # Metric-related quantity for Armijo
        m = metric_Q(Gk, eta_k, Qk, 1)
        m = min(m, -1e-14)
        t = min(alpha_bar, 1 / grad_norm_arr[k])

        # ------------------ Backtracking line search ------------------
        E_Qk =  E_Q_arr[k]
        Q_trial = Qk
        E_trial = E_Q_arr[k]
        while true
            Q_trial = Retr(Qk, t * eta_k)
            E_trial = E(Q_trial, Zmat)

            # Armijo condition
            if E_trial <= E_Qk + c * t * m && isposdef(Q_trial)
                break
            else
                t *= beta
                if t < 1e-16  # stepsize too small
                    break
                end
            end
        end
        #println("final step-size :", t)

        # Update iterate if Q_trial is positive definite, otherwise keep Qk and break to avoid numerical issues
        if isposdef(Q_trial)
            Qk = Q_trial
            E_Q_arr[k+1] = E_trial
            Gk = grad_E(Qk, Zmat)
            grad_norm_arr[k+1] = sqrt(abs(norm_Q(Gk, Qk, metric_type)))
        else
            println("Warning: Obtained non-positive definite matrix at iteration $k, stopping optimization.")
            iter = maxIter # to indicate no convergence
            break
        end

        # stopping criterion for non-increasing gradient norm
        if norm_grad_stagnation(grad_norm_arr, k, 10, 1e-8)
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
        if k == 1 || k % 20 == 0
            println("Iter $k | E_Q = $(round(E_Q_arr[k], sigdigits=4))| grad norm = $(round(grad_norm_arr[k], sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors")
        end
        iter = iter + 1
    end

    return Qk, E_Q_arr[1:iter], grad_norm_arr[1:iter], iter
end

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

function norm_Q(A::AbstractMatrix{<:Real}, 
                Q::AbstractMatrix{<:Real}, 
                metric_type::Int)

    if metric_type == 1
        # Affine-invariant metric: ⟨A,A⟩_Q = tr((Q⁻¹A)(Q⁻¹A))

        try
            F = cholesky(Symmetric(Q))   # more stable than Q\A
            X = F \ A                    # solves Q * X = A
            val = tr(X * X)

        catch err
            if err isa LinearAlgebra.SingularException
                # Fallback: regularize Q slightly
                ϵ = 1e-12
                F = cholesky(Symmetric(Q + ϵ*I))
                X = F \ A
                val = tr(X * X)
            else
                rethrow(err)
            end
        end

    else
        # Euclidean metric
        val = tr(A * A)
        if val < 0
            val = abs(val)
        end
    end

    return val
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


end # module