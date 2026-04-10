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
# Run RTR on multiple initial Qs
# --------------------------------------------
function run_multiple_RTR(Qs,Z_universal, n::Int, C::Real; tol, maxIter)
    total = length(Qs)
    progress = Atomic{Int}(0)
    
    N = length(Qs)
    Qopts = Vector{Matrix{Float64}}(undef, N)
    energies = zeros(N)
    iterations = zeros(Int, N)
    infos = Vector{Any}(undef, N)

    Threads.@threads for i in 1:N
        #println("Optimizing from initial matrix $i/$N")
        Q0 = Qs[i]
        #Zmat = enumerate_quadint(Q0, C)
        result =  run_Riemannian_TR(Q0, Z_universal, alpha, C, maxIter, tol)

        # store results
        Qopts[i] = result.Q_opt
        iterations[i] = result.iter
        energies[i] = result.E_arr[end]
        infos[i] = result.info

        done = atomic_add!(progress, 1) + 1
        println("Completed $done / $total (thread $(threadid()))")
    end

    # Normalize and reduce final Qopts
    function normalize_and_reduce(Q)
        B = Matrix(cholesky(Q).U)
        B = lll(B)[1]
        Qr = B' * B
        return Qr / det(Qr)^(1/size(Q,1))
    end
    Qopts = normalize_and_reduce.(Qopts)

    return Qopts, energies, iterations, infos
end

# ---------------------- Run RTR on multible initial Q_0 ----------------------
# idea to find multible local minimausing N random initial Qs.

# Radii used for the universal PEL, depending on the dimension n.
#These are chosen based on preliminary experiments to balance computational cost
# and quality of results.
function radius_C(n)
    if n==2
        C = 500
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


n = 2                       # dimension of the lattice
alpha =  float(pi)          # parameter alpha in the energy functional
Method = "RTR_"             # Method used
take_Identity_as_starting_point = false
N = 10 # number of initial Qs
C = radius_C(n) 
println("Using C = $C")
metric_type = 1 # affine-invariant metric

#---------------------------load initial Gram matrices------------------------------------------
folder_initial_matrices = joinpath(input_data, "Initial_Gram_matrices/")
@load joinpath(folder_initial_matrices,"Random_initial_Qs_n$(n).jld2") Qs
Qs = Qs[1:N]         

# If the partial evaluation lattice should be updated at each iteration, set evaluationUpdate to true.
# If not, set to fals. Better convergence if it is not updated. 
evaluationUpdate = false

# Energy, gradient, retraction and evaluation matrix Zmat
E = (Q, Z) -> compute_E_energy(Q, alpha, Z)
grad_E = (Q, Z) -> Riemannian_grad(Q, alpha, Z)
Retr = (Q, H) -> exponentialRetraction(Q, H)
Qref = Matrix{Float64}(I, n, n)

#------------------------Load universal PEM-------------------------------------------------
folder_PEM = joinpath(input_data, "Partial_Evaluation_Lattices/")
@load joinpath(folder_PEM, "universal_PEL_n$(n)_C$(C).jld2") PEM
Z_universal = PEM

#------------------------------Run Optmizations-----------------------------------------------
maxIter = 50
Qopts, energies, iterations, _ =  run_multiple_RTR(Qs, Z_universal, n, C; tol = 5e-12, maxIter= maxIter) 

#-------------------------------- only keep Qs with iterations < maxIter-----------------------
keep_idx = findall(iterations .< maxIter)
Qopts    = Qopts[keep_idx]
energies = energies[keep_idx]
println("Kept $(length(Qopts)) entries after removing minimizers with max iterations")

#------------------------------------only keep unique Qopts------------------------------------
#compare via fro norm
function relative_frobenius_diff_sym(A::AbstractMatrix, B::AbstractMatrix; symmetric::Bool=false)
    if symmetric
        n = size(A,1)
        @assert size(A) == size(B) "Matrices must have the same size"
        diff_sq = 0.0
        normA_sq = 0.0
        for i in 1:n
            for j in i:n
                w = i==j ? 1.0 : 2.0
                diff_sq += w*(A[i,j]-B[i,j])^2
                normA_sq += w*(A[i,j])^2
            end
        end
        rel_diff = sqrt(diff_sq)/sqrt(normA_sq)
        return rel_diff
    else
        return norm(A-B, fro)/norm(A, fro)
    end
end
function unique_matrices(Qs::Vector{<:AbstractMatrix}, energies::Vector; tol::Float64=1e-8)
    
    keep_idxs = Int[]

    for i in 1:length(Qs)
        is_duplicate = false
        for j in keep_idxs
            rel_diff = relative_frobenius_diff_sym(Qs[i], Qs[j], symmetric=true)
            #println("diff", rel_diff)
            if rel_diff < tol
                is_duplicate = true
                break
            end
        end
        if !is_duplicate
            push!(keep_idxs, i)
        end
    end

    return Qs[keep_idxs], energies[keep_idxs]
end

Qopts, energies = unique_matrices(Qopts, energies)
println("Kept $(length(Qopts)) entries after removing duplicates")

#------------------------------ test for local minimizers---------------------------------------------
  # test if each Qopt is second-order critical point, i.e. local minimizer.
@assert length(Qopts) == length(energies)

is_local_min = falses(length(Qopts))
length_Qopts_old = length(Qopts)
Threads.@threads for i in eachindex(Qopts)
   Zmat = truncate_Universal_Evaluation_Matrix_fast(Qopts[i], Z_universal, alpha, 1e-50)
    result, λ_min,_ = is_second_order_critical(Qopts[i], alpha, Zmat)
    is_local_min[i] = result === true || result == true
    println("$(i) of $(eachindex(Qopts)) tested for local minimizer with λ_min=", round(λ_min, sigdigits=6))
end

Qopts      = Qopts[is_local_min]
energies   = energies[is_local_min]

println("Kept $(length(Qopts)) / $(length_Qopts_old) potential local minimizers.")


# ------------------------Save results to JLD2 and text file--------------------------------------
 function save_matrices_txt(filename, Qs)
    open(filename, "w") do io
        for (k, Q) in enumerate(Qs)
            println(io, "# Matrix ", k)
            for i in 1:size(Q,1)
                println(io, join(Q[i, :], " "))
            end
            println(io)  # blank line
        end
    end
end

@save joinpath(results_dir,"results_$(Method)alphaPI","n_$(n)" , "$(Method)optimized_Qs_n$(n)_alphaPI_test.jld2") Qopts energies
save_matrices_txt(joinpath(results_dir,"results_$(Method)alphaPI","n_$(n)" , "$(Method)optimized_Qs_n$(n)_alphaPI_test.txt"), Qopts)

