module RiemannianTrustRegionMethod

export Riemannian_tCG, RTR_optimizer, tangentProjection, RTR_optimizerNC, truncate_Universal_Evaluation_Matrix, truncate_Universal_Evaluation_Matrix_fast, norm_grad_non_increasing, no_decrease_condition


using LinearAlgebra
include("EvaluationMatrix.jl")
include("GeometricObjects.jl")
using .GeometricObjects: compute_E_energy, Riemannian_grad, exponentialRetraction, Riemannian_Hessian
using .EvaluationMatrix: enumerate_quadint
using Base.Threads
using JLD2
using LoopVectorization
using Statistics
#using Revise

#old optimizer
function RTR_optimizer(Q0, Zmat, E, grad_E, hess_E, Retr;
                       Delta0=0.1, eta=0.1, maxCG=150, maxIter=1000, tol=1e-12, evaluationUpdate = false,
                       opts=Dict())
    Q = copy(Q0)
    Delta = Delta0

    # Preallocate info array
    info = Vector{Dict}(undef, maxIter+1)
    E_Q_arr = zeros(maxIter+1)
    E_Q_arr[1] = E(Q0, Zmat)
    grad_norm_arr = zeros(maxIter+1)
    G = grad_E(Q0, Zmat)
    grad_norm_arr[1] = sqrt(tr((Q0 \ G) * (Q0 \ G)'))

    evaluationSize_1 = size(Zmat, 1)
    evaluationSize_2 = 0
    iter_global = 1

    for iter in 1:maxIter
        # Stopping condition (before the first iteration, Q0 could be optimal)
        if grad_norm_arr[iter] < tol && no_decrease_condition(E_Q_arr, tol)
            println("* Converged at iter $iter | E_Q=$(E(Q,Zmat)) | grad norm=$(grad_norm_arr[iter]) | Evaluates over $(size(Zmat, 1)) vectors " )
            break
        end

        if iter >=2 && evaluationUpdate
            evaluationSize_1 = evaluationSize_2
        end
        # Gradient and its Riemannian norm
        G = grad_E(Q, Zmat)
        gradNorm2 = tr((Q \ G) * (Q \ G)')  # squared Riemannian norm
        grad_norm_arr[iter+1] = sqrt(gradNorm2)

        # Evaluate current cost
        E_Q = E(Q, Zmat)
        E_Q_arr[iter+1] = E_Q

        # Solve trust-region subproblem (tCG)
        H, flag = Riemannian_tCG(Q, Zmat, grad_E, hess_E, Delta, eta, maxCG)
        #Riemannian_tCG(Q, Zmat, E, grad_E, hess_E, Delta, eta, maxCG)

        # Retraction to get new point
        Q_new = Retr(Q, H)

        # Compute actual and predicted reduction
        Hv = hess_E(Q, H, Zmat)
        H_proj = tangentProjection(H, Q)
        Hv_proj = tangentProjection(Hv, Q)

        predRed = -tr((Q \ G)*(Q \ H)') - 0.5*tr((Q \ H_proj)*(Q \ Hv_proj)')
        actRed = E_Q - E(Q_new, Zmat)
        rho = actRed / predRed

        # Update trust-region radius
        if rho < 0.25
            Delta *= 0.25
        elseif rho > 0.75 && abs(norm(H_proj) - Delta) < 1e-8
            Delta = min(2*Delta, 100.0)
        end

        # Accept or reject step
        if rho > 0
            Q = Q_new
        end
        iter_global  = iter_global +1
        # Store iteration info
        info[iter + 1] = Dict(
            :iter => iter,
            :gradNorm => gradNorm2,
            :actRed => actRed,
            :predRed => predRed,
            :Delta => Delta,
            :flag => flag,
            :E_alpha => E_Q
        )

        # Display progress every 10 iterations
        if iter % 10 == 0 || iter == 1
            println("|Iter $iter| E_Q=$(E(Q_new,Zmat)) | grad norm=$(grad_norm_arr[iter]) | Evaluates over $(size(Zmat, 1)) vectors ")
        end

       #Adapation of EvaluationMatrix
        if evaluationUpdate
          Zmat = enumerate_quadint(Q, opts[:C]) # update Zmat if needed
          evaluationSize_2 = size(Zmat, 1)
        end
        if iter >= 2 && evaluationUpdate
            if evaluationSize_1 == evaluationSize_2
                evaluationUpdate = false # stop updating if size didn't change
            end
        end
    end

    Q_opt = Q
    info = info[1:iter_global]
    E_Q_arr = E_Q_arr[1:iter_global]
    grad_norm_arr = grad_norm_arr[1:iter_global]

    return Q_opt, info, E_Q_arr, grad_norm_arr, iter_global
end

"""
truncate the universal evaluation matrix by keeping only rows where e^-α Q[z] >= threshold or
by summands α e^-α Q[z] Q[z] >= threshold for the gradient
Goal: reduce the size of the evaluation matrix while keeping the most relevant vectors for the energy and gradient evaluation
"""
function truncate_Universal_Evaluation_Matrix(Q, Zmat, alpha, threshold)
    U = cholesky(Q).U
    eval_values = exp.(-alpha .* vec(sum((Zmat * U).^2, dims=2)))
    mask = eval_values .>= threshold
    return Zmat[mask, :]
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

function truncate_Universal_Evaluation_Matrix_fast(Q, Zmat, alpha, threshold)
    U = cholesky(Q).U
    ZU = Zmat * U

    n, m = size(ZU)
    mask = Vector{Bool}(undef, n)

    @threads for i in 1:n
        s = 0.0
        @turbo for j in 1:m
            v = ZU[i, j]
            s += v * v
        end
        mask[i] = exp(-alpha * s) >= threshold
    end

    return Zmat[mask, :]
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
Riemannian truncated Conjugate Gradient (tCG) for the trust-region subproblem.

Inputs:
- Q        :: SPD matrix (n×n), current point on manifold
- Zmat     :: evaluation matrix
- grad_E   :: function grad_E(Q, Zmat)
- hess_E   :: function hess_E(Q, η, Zmat)
- Delta    :: trust-region radius
- eta      :: inexact Newton tolerance
- maxIter  :: maximum number of CG iterations

Sequences:
- r_j      :: residuals. Error to the Newton Equation
- p_j      :: span a subspace of T_Q P
- 

Outputs:
- η        :: tangent vector solving TR subproblem approximately
- flag     :: termination reason
              0 = residual small
              1 = trust-region boundary hit
"""

function Riemannian_tCG(
    Q::AbstractMatrix,
    Zmat,
    G,
    hess_E,
    Delta::Real,
    eta::Real,
    maxIter::Int
)
    # Riemannian gradient
    #G = grad_E(Q, Zmat)

    # Initial residual (projected gradient)
    r = -tangentProjection(G, Q)
    p = copy(r)
    η = zeros(size(Q))

    # Norm in metric <X,Y>_Q = tr((Q\X)(Q\Y)')
    function inner_Q(X, Y)
        return tr((Q \ X) * (Q \ Y)')
    end

    norm_r0 = sqrt(inner_Q(r, r))
    flag = 0

    for k in 1:maxIter
        # Hessian-vector product
        Hp = hess_E(Q, p, Zmat)
        Hp = tangentProjection(Hp, Q)

        inner_r_r  = inner_Q(r, r)
        inner_p_Hp = inner_Q(p, Hp)

        # Negative curvature
        if inner_p_Hp <= 0
            τ = compute_tau(η, p, Delta, Q)
            η += τ * p
            flag = 1
            break
        end

        α = inner_r_r / inner_p_Hp
        η_new = η + α * p

        # Trust-region boundary check
        if sqrt(inner_Q(η_new, η_new)) >= Delta
            τ = compute_tau(η, p, Delta, Q)
           η += τ * p
            flag = 1
            break
        end

        η = η_new

        # Residual update
        r_new = r - α * Hp
        r_new = tangentProjection(r_new, Q)

        norm_r = sqrt(inner_Q(r_new, r_new))
        if norm_r <= eta * norm_r0
            r = r_new
            break
        end

        # Polak–Ribiere update
        beta = inner_Q(r_new, r_new) / inner_r_r
        #p = r_new + beta * p
        p = tangentProjection(r_new + beta * p, Q)
        r = r_new
    end

    return η, flag
end

"""
Compute tau such that ||η + tau*p||_Q = Delta
"""
function compute_tau(
    η::AbstractMatrix,
    p::AbstractMatrix,
    Delta::Real,
    Q::AbstractMatrix
)
    inner_Q(X, Y) = tr((Q \ X) * (Q \ Y)')

    a = inner_Q(p, p)
    b = 2 * inner_Q(η, p)
    c = inner_Q(η, η) - Delta^2

    disc = b^2 - 4a*c
    if disc < 0
        return 0.0  # fallback (should not happen)
    end

    sqrt_disc = sqrt(disc)
    tau1 = (-b + sqrt_disc) / (2a)
    tau2 = (-b - sqrt_disc) / (2a)

    return max(tau1, tau2)
end


"""
Riemannian Trust-Region optimizer (RTR) for SPD matrices with det=1.

Inputs:
- Q0      : n×n SPD matrix (det=1)
- Zmat    : k×n integer matrix for function evaluation
- E       : cost function E(Q, Zmat)
- grad_E  : gradient function grad_E(Q, Zmat)
- hess_E  : Hessian-vector product hess_E(Q, \eta, Zmat)
- Retr    : retraction function Retr(Q, \eta)
- Delta0  : initial trust-region radius
- eta     : inexact Newton tolerance for tCG
- maxCG   : max truncated CG iterations per RTR step
- maxIter : max RTR iterations
- tol     : stopping criterion
- opts    : Dict with options (e.g., :alpha)

Outputs:
- Q_opt       : optimized SPD matrix
- info        : array of iteration info (structs)
- E_Q_arr     : cost function history
- grad_norm_arr : Riemannian gradient norms
- iter        : number of iterations performed
"""


"""
#test truncation of universal evaluation matrix
n = 6
C = 20
Q = Matrix{Float64}(I, n, n)
alpha = float(pi)
folder_PEM = "Partial_Evaluation_Lattices/"
@load joinpath(folder_PEM, "universal_PEL_n(n)_C(C).jld2") PEM
Z_universal = PEM
# comapre speed of two truncation functions
@time Z_truncated_fast = truncate_Universal_Evaluation_Matrix_fast(Q, Z_universal, alpha, 1e-40)
@time Z_truncated = truncate_Universal_Evaluation_Matrix(Q, Z_universal, alpha, 1e-40)


Riemannian Trust Region Method fixing the issue with incresing norm grad
"""
function RTR_optimizerNC(Q0, Zmat, E, grad_E, hess_E, Retr;
                       Delta0=0.1, eta=0.1, maxCG=50, maxIter=1000, tol=1e-12, evaluationUpdate = false,
                       opts=Dict())
    Q = copy(Q0)
    Delta = Delta0
    Zmat_original = copy(Zmat)
    # Preallocate info array
    info = Vector{Dict}(undef, maxIter+1)
    setprecision(256)
    E_Q_arr = zeros(maxIter+1)
    E_Q_arr[1] = E(Q0, Zmat)
    grad_norm_arr = zeros(maxIter+1)
    G = grad_E(Q0, Zmat)
    grad_norm_arr[1] = sqrt(tr((Q0 \ G) * (Q0 \ G)'))


    evaluationSize_1 = size(Zmat, 1)
    evaluationSize_2 = 0
    iter_global = 1

    for iter in 1:maxIter

        # truncate universal evaluation matrix for values e^-α Q[z] zz^T< 1e-50
        if iter % 25 == 0
        #Zmat = truncate_Evaluation_Matrix_gradient(Q, Zmat_original, opts[:alpha], 1e-80) #1e-100
        #Zmat = truncate_Universal_Evaluation_Matrix_fast(Q, Zmat_original, opts[:alpha], 1e-40)
        end
        # Stopping condition (before the first iteration, Q0 could be optimal)
        
        if grad_norm_arr[iter] < tol
            println("*** Converged at iter $iter | E_Q=$(round.(E_Q_arr[iter]; sigdigits=4)) | grad norm=$(round.(grad_norm_arr[iter]; sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors " )
            break
        end
        if iter >=2 && evaluationUpdate
            evaluationSize_1 = evaluationSize_2
        end

        G = grad_E(Q,Zmat)
        # Solve trust-region subproblem (tCG)
        η, _ = Riemannian_tCG(Q, Zmat, G, hess_E, Delta, eta, maxCG)
        #Riemannian_tCG(Q, Zmat, E, grad_E, hess_E, Delta, eta, maxCG)

        # Compute actual and predicted reduction
        η = tangentProjection(η, Q)
        #H = hess_E(Q, η, Zmat)
        #H = tangentProjection(H, Q)
        predRed = -  0.5 * tr((Q \ G)*(Q \ η)') # - 0.5*tr((Q \ η)*(Q \ H)')

        # Retraction to get new point
        Q_new = Retr(Q, η)
        E_trial = E(Q_new, Zmat)
        actRed = E_Q_arr[iter] - E_trial #actual reduction is often zero
        #println("Actual reduction at iter $iter: ", actRed)
        rho = actRed / max(predRed, eps())

        # Update trust-region radius
        if rho < 0.25
            #Delta *= 0.25 #0.25 scales the trust region radius down 
            Delta = max(0.1 *Delta, 1e-6)
        elseif rho > 0.75 && abs(norm(η) - Delta) < 1e-8
            Delta = min(2*Delta, 10.0)
        end

        iter_global  = iter_global + 1
        #println("rho at iter $iter: ", rho)
        
        # Accept or reject step
        rho_prime = 0.1 # in [0,0.25)
        if rho > rho_prime # is not entered for certain initical matrices, if entered rho is most of the time the same
            Q = Q_new
            E_Q_arr[iter_global] = E_trial
            G = grad_E(Q, Zmat)
            grad_trial_Norm =sqrt( tr((Q \ G) * (Q \ G)'))
            grad_norm_arr[iter_global] = grad_trial_Norm
        else
            E_Q_arr[iter_global] = E_Q_arr[iter] # keep old energy
            grad_norm_arr[iter_global] = grad_norm_arr[iter] # keep old gradient norm
        end

        # stopping criterion for non-increasing gradient norm
        if norm_grad_stagnation(grad_norm_arr, iter, 5, 1e-10)
             println(":-( No decrease in energy at iter $iter | E_Q=$(round(E_Q_arr[iter], sigdigits=4)) | grad norm=$(round(grad_norm_arr[iter], sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors " )
             iter_global = maxIter # to indicate no convergence
             break
        end

        #println("Delta = ", Delta)
        norm_eta = sqrt(tr((Q \ η)*(Q \ η)'))
        #println("||η|| = ", norm_eta)
        #println(" distance of Delta and ||η||:", abs(norm_eta-Delta))
        
        # Display progress every 10 iterations
        if iter % 100== 0 || iter == 100
            println("|Iter $iter| E_Q=$(round(E_Q_arr[iter_global], sigdigits=4)) | grad norm=$(round(grad_norm_arr[iter_global], sigdigits=4)) | Evaluates over $(size(Zmat, 1)) vectors ")
        end

       #Adapation of EvaluationMatrix
        if evaluationUpdate
          Zmat = enumerate_quadint(Q, opts[:C]) # update Zmat if needed
          evaluationSize_2 = size(Zmat, 1)
        end
        if iter >= 2 && evaluationUpdate
            if evaluationSize_1 == evaluationSize_2
                evaluationUpdate = false # stop updating if size didn't change
            end
        end
    end

    Q_opt = Q
    info = info[1:iter_global]
    E_Q_arr = E_Q_arr[1:iter_global]
    grad_norm_arr = grad_norm_arr[1:iter_global]

    return Q_opt, info, E_Q_arr, grad_norm_arr, iter_global
end

function norm_grad_non_increasing(norm_G_new, norm_G_old)
    non_increasing =( norm_G_new <= norm_G_old)
    return non_increasing
end

function no_decrease_condition(function_values::AbstractVector{<:Real}, tol::Real)
    # Find indices of non-zero entries
    idx = findall(!=(0), function_values)
    
    # Need at least two non-zero values
    if length(idx) < 20
        return false
    end
    
    # Take the last two non-zero indices
    i1, i2 = idx[end-19], idx[end]
    # Relative difference
    b = abs(function_values[i2] - function_values[i1]) / min(1.0, abs(function_values[i1])) < tol
    return b
end



# test second order model correctness
function test_second_order_model(Q, η, α, Zmat)
    E0 = compute_E_energy(Q, α, Zmat)    #E_Qk
    G = Riemannian_grad(Q, α, Zmat)      #G_Qk
    Hess = Riemannian_Hessian(Q, η, α, Zmat)   #Hess_Q_k

    # model prediction
    pred = E0 + tr((Q \ G)' * (Q \ η)) + 0.5 * tr((Q \ η)' * (Q \ Hess)) #2nd order model
    # actual energy at retracted point
    Q_new = exponentialRetraction(Q, η)
    E_new = compute_E_energy(Q_new, α, Zmat)

    println("Predicted energy: ", pred)
    println("Actual energy: ", E_new)
    println("Difference: ", abs(pred - E_new))
end



"""
n = 8 # dimension of the problem
alpha = float(pi) # parameter alpha in the energy functional
N = 10 # number of initial Qs
Method = "RTR_"
C = 12 
println("Using C = C")

metric_type = 1 # affine-invariant metric

folder_name = "results_(Method)alphaPI/"
@load joinpath(folder_name,"(Method)initial_Qs_n(n)_alphaPI.jld2") Qs
Qs = Qs[1:N]

# Do not update PEL, instead a universal PEL is used
evaluationUpdate = false

# Energy, gradient, retraction and evaluation matrix Zmat
E = (Q, Z) -> compute_E_energy(Q, alpha, Z)
grad_E = (Q, Z) -> Riemannian_grad(Q, alpha, Z)
Retr = (Q, H) -> exponentialRetraction(Q, H)
hess_E(Q, H, Z) = Riemannian_Hessian(Q, H, alpha, Z) 
Delta = 1e-8   #determining parameter for the model fit
eta = 0.001
maxCG = 50

#------------------------Load universal PEM-------------------------------------------------
function truncate_Universal_Evaluation_Matrix(Q, Zmat, alpha, threshold)
    U = cholesky(Q).U
    eval_values = exp.(-alpha .* vec(sum((Zmat * U).^2, dims=2)))
    mask = eval_values .>= threshold
    return Zmat[mask, :]
end

folder_PEM = "Partial_Evaluation_Lattices/"
@load joinpath(folder_PEM, "universal_PEL_n(n)_C(C).jld2") PEM
Z_universal = PEM
Q = Qs[8]
Zmat = truncate_Universal_Evaluation_Matrix(Q, Z_universal, alpha, 1e-50)
println("Size of truncated Zmat: ", size(Zmat))

η, _ = Riemannian_tCG(Q, Zmat, grad_E, hess_E, Delta, eta, maxCG)
test_second_order_model(Q, η, alpha, Zmat)
"""


end # module