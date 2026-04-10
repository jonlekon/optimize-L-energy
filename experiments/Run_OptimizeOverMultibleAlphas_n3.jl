#works
using LinearAlgebra
using LLLplus
using Random
using JLD2
using DelimitedFiles
using Base.Threads
using Revise
using Base.Threads
using FLoops

src = joinpath(@__DIR__, "..", "src")
input_data = joinpath(@__DIR__, "..", "input_data")
results_dir = joinpath(@__DIR__, "..", "results")

include(joinpath(src, "EvaluationMatrix.jl"))
include(joinpath(src, "GeometricObjects.jl"))
include(joinpath(src, "module RiemannianTrustRegionMethod.jl"))
include(joinpath(src, "module second_order_critical.jl"))

using .EvaluationMatrix: enumerate_quadint
using .GeometricObjects: compute_E_energy, Riemannian_grad, exponentialRetraction, Riemannian_Hessian
using .RiemannianTrustRegionMethod: RTR_optimizer, RTR_optimizerNC, truncate_Universal_Evaluation_Matrix, truncate_Universal_Evaluation_Matrix_fast, norm_grad_non_increasing, no_decrease_condition
using .second_order_critical: is_second_order_critical
#using .CleanGramMatrices: lattice_isometry

println("Number of threads: ", nthreads())

# --------------------------------------------
# Run Riemannian Trust-Region (RTR) for one Q0
# --------------------------------------------
function run_Riemannian_TR(Q0,
    Z_universal,
    alpha::Real,
    C::Real,
    maxIter::Int ,
    tol::Real;
    Delta0::Real =Delta = 0.9 , # initial trust-region radius. 
    eta::Real = 0.0001, # scalar tells how much the Newton Equation is exactly solved 
    maxCG::Int = 50,
    evaluationUpdate::Bool = false,
    opts::Dict = Dict()
)

    # ---------------------- Enumerate integer vectors ----------------------
    Zmat = Z_universal #enumerate_quadint(Q0, C)
    B0 = cholesky(Q0).U
    # ---------------------- Energy, gradient, Hessian, retraction ----------------------
    E(Q, Z)      = compute_E_energy(Q, alpha, Z)
    grad_E(Q, Z) = Riemannian_grad(Q, alpha, Z)
    Retr(Q, H)   = exponentialRetraction(Q, H)
    hess_E(Q, H, Z) = Riemannian_Hessian(Q, H, alpha, Z)  # you need to implement this

    # ---------------------- Default options ----------------------
    default_opts = Dict(
        :alpha_bar => 1.0,
        :C         => C,
        :alpha      => alpha,
    )
    for (k,v) in opts
        default_opts[k] = v
    end

    # ---------------------- Run RTR optimizer ----------------------
    Q_opt, info, E_arr, grad_arr, iter = RTR_optimizerNC(
        Q0, Zmat, E, grad_E, hess_E, Retr;
        Delta0=Delta0, eta=eta, maxCG=maxCG, maxIter=maxIter, tol=tol, evaluationUpdate=evaluationUpdate ,opts=default_opts
    )

    # ---------------------- Extract reduced basis ----------------------
    B_opt = lll(Matrix(cholesky(Q_opt).U))[1]

    return (
        Q_opt=Q_opt,
        B_opt=B_opt,
        B0=B0,
        E_arr=E_arr,
        grad_arr=grad_arr,
        iter=iter,
        info=info
    )
end

# --------------------------------------------
# Run RTR on multiple alphas and multiple initial Qs
# --------------------------------------------
function run_multiple_RTR(Qs, Z_universal, n::Int, alphas, C::Real; tol, maxIter)
    N1 = length(alphas)
    N2 = length(Qs)
    total = N1 * N2
    Qopts = Vector{Matrix{Float64}}(undef, total)
    energies = zeros(total)
    iterations = zeros(Int, total)
    alphas_out = zeros(total)
    progress = Atomic{Int}(0)
    initial_matrix_number = zeros(Int, total)

    Threads.@threads for i in 1:N1
        for j in 1:N2
            idx = (i-1)*N2 + j
            alpha = alphas[i]
           # println("alpha: ", alpha)
            result = run_Riemannian_TR(Qs[j], Z_universal, alpha, C, maxIter, tol)
            # store results
            Qopts[idx] = result.Q_opt
            iterations[idx] = result.iter
            energies[idx] = result.E_arr[end]
            alphas_out[idx] = alpha
            done = atomic_add!(progress, 1) + 1 
            println("Completed $done / $total (thread $(threadid()))")
            initial_matrix_number[idx] = j
        end
        
    end

    # Normalize and reduce final Qopts
    function normalize_and_reduce(Q)
        B = Matrix(cholesky(Q).U)
        B = lll(B)[1]
        Qr = B' * B
        return Qr / det(Qr)^(1/size(Q,1))
    end
    Qopts = normalize_and_reduce.(Qopts)

    return Qopts, energies, iterations, alphas_out, initial_matrix_number
end

# ---------------------- Run RTR on multible alphas ----------------------

function save_matrices_txt(filename, Qs, energies, alphas, lambda, initial_matrix_number)
    open(filename, "w") do io
        for (k, Q) in enumerate(Qs)
            println(io, "# Matrix ", initial_matrix_number[k], "| For alpha = ", alphas[k])
            for i in 1:size(Q,1)
                println(io, join(round.(Q[i, :]; sigdigits=4), " ")) # round.(energies[k]; sigdigits=4)
            end
            println(io)  # blank line
            println(io, "Energy = ", round.(energies[k]; sigdigits=4))
            println(io)  # blank line
            println(io, "Eigenvalues of the Hessian matrix :")
            println(io, join(round.(lambda[k, :]; sigdigits=4), " "))
            println(io)  # blank line
        end
    end
end


n = 3 # dimension of the lattice
N = 100 # number of initial Qs
#alphas = collect(0.4:0.002:0.45) #alpha_critical in [0.45, 0.5 ]
alphas = vcat(
    collect(0:0.5:20),     # very fine near transition
)
Method = "RTR_"
C = 100
println("Using C = $C")
#---------------------------load  initial matrices------------------------------------------
folder_initial_matrices = joinpath(input_data, "Initial_Gram_matrices/")
@load joinpath(folder_initial_matrices,"Random_initial_Qs_n$(n).jld2") Qs
Qs = Qs[1:N]  
Q_FCC = 2^(-2/3)*[2 -1 -1;
                  -1 2  1;
                  -1  1 2  ] 
FCC_dual = 2^(-4/3)*[3 1 1
                    1 3 -1
                    1 -1 3]    
metric_type = 1 # affine-invariant metric
evaluationUpdate = false

#------------------------Load universal PEM-------------------------------------------------
folder_PEM = joinpath(input_data, "Partial_Evaluation_Lattices/")
@load joinpath(folder_PEM, "universal_PEL_n$(n)_C$(C).jld2") PEM
Z_universal = PEM

#------------------------------Run Optmization-----------------------------------------------
maxIter = 50
Qopts, energies, iterations, alphas, initial_matrix_number =  run_multiple_RTR(Qs, Z_universal, n, alphas, C; tol = 5e-10, maxIter= maxIter) 
#-------------------------------- only keep Qs with iterations < maxIter-----------------------
keep_idx = findall(iterations .< maxIter)
Qopts    = Qopts[keep_idx]
energies = energies[keep_idx]
alphas   = alphas[keep_idx]
initial_matrix_number = initial_matrix_number[keep_idx]
println("Kept $(size(Qopts)) entries after removing minimizers with max iterations")

temp_folder = joinpath(results_dir, "alpha_critical_analysis/", "temporary_results/")
@save joinpath(temp_folder, "checkpoint_Qopts.jld2") Qopts
@save joinpath(temp_folder, "checkpoint_energies.jld2") energies
@save joinpath(temp_folder, "checkpoint_alphas.jld2") alphas
@save joinpath(temp_folder, "checkpoint_initial_matrix_number.jld2") initial_matrix_number

#------------------------------ test for local minimizers---------------------------------------------
@assert length(Qopts) == length(energies)
progress = Atomic{Int}(0)
d = n*(n+1) ÷ 2 
lambda =  zeros(length(Qopts), (d-1)) 
is_local_min = falses(length(Qopts))
length_Qopts_old = length(Qopts)

Threads.@threads for i in eachindex(Qopts)
    Zmat = enumerate_quadint(Qopts[i], 100)
    result, λ_min,EVs = is_second_order_critical(Qopts[i], alphas[i], Zmat)
    is_local_min[i] = result === true || result == true
    done = atomic_add!(progress, 1) + 1 
    println("Completed $done / $length_Qopts_old (thread $(threadid()))", "| λ_min = ", round(λ_min; sigdigits=4))
    lambda[i, :] = EVs'
end

Qopts      = Qopts[is_local_min]
energies   = energies[is_local_min]
alphas     = alphas[is_local_min]
lambda     = lambda[is_local_min, :]
initial_matrix_number = initial_matrix_number[is_local_min]

println("Kept $(length(Qopts)) / $(length_Qopts_old) local minimizers.")

@save joinpath(temp_folder, "checkpoint_Qopts.jld2") Qopts
@save joinpath(temp_folder, "checkpoint_energies.jld2") energies
@save joinpath(temp_folder, "checkpoint_alphas.jld2") alphas
@save joinpath(temp_folder, "checkpoint_initial_matrix_number.jld2") initial_matrix_number
@save joinpath(temp_folder, "checkpoint_lambda.jld2") lambda
