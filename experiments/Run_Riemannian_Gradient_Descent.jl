#works
using LinearAlgebra
using LLLplus
using Random
using JLD2
using DelimitedFiles
using Base.Threads
using FLoops
using Revise

src = joinpath(@__DIR__, "..", "src")
input_data = joinpath(@__DIR__, "..", "input_data")
results_dir = joinpath(@__DIR__, "..", "results")

include(joinpath(src, "EvaluationMatrix.jl"))
include(joinpath(src, "GeometricObjects.jl"))
include(joinpath(src, "module RiemannianGradientDescent.jl"))
include(joinpath(src, "module OptimalLattices.jl"))
include(joinpath(src, "module second_order_critical.jl"))

using .EvaluationMatrix: enumerate_quadint
using .GeometricObjects: compute_E_energy,
                         Riemannian_grad,
                         exponentialRetraction
using .RiemannianGradientDescent: AcceleratedLineSearch_P
using .second_order_critical: is_second_order_critical

println("Number of threads: ", nthreads())

# ---------------------- Run Riemannian gradient descent on multible initial Q_0 ----------------------
# idea to find multible local minimausing N random initial Qs.

# Save Gram matrices to text file
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

# Use LLL basis reduction to make lattices comparable.
function normalize_and_reduce(Q)
    B = Matrix(cholesky(Q).U)
    B = lll(B)[1]
    Qr = B' * B
    return Qr / det(Qr)^(1/size(Q,1))
end

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

#------------------- Parameters for optimization ----------------------

n = 2           #dimension of the lattice
Method = "RGD_" #Method used
N = 10       # number of initial Qs
alpha = float(pi) # parameter alpha in the energy functional
C = radius_C(n)
println("Using radius C = $C")

metric_type = 1 # 1 = affine-invariant metric <A,B>_Q = tr(Q\A*Q\B)

# If the partial evaluation lattice should be updated at each iteration, set evaluationUpdate to true.
# If not, set to fals. Better convergence if it is not updated. 
evaluationUpdate = false #no internal update of PEM

maxIter = 60 # maximum number of iterations for the optimization

#------------------Load initial Qs from file-------------------------------
folder_initial_matrices = joinpath(input_data, "Initial_Gram_matrices/")
@load joinpath(folder_initial_matrices,"Random_initial_Qs_n$(n).jld2") Qs
Qs = Qs[1:N]

#------------------------Load universal PEM-------------------------------------------------
folder_PEM = joinpath(input_data, "Partial_Evaluation_Lattices/")
@load joinpath(folder_PEM, "universal_PEL_n$(n)_C$(C).jld2") PEM
Z_universal = PEM

# Energy, gradient, retraction and evaluation matrix Z_universal
E = (Q, Z) -> compute_E_energy(Q, alpha, Z)
grad_E = (Q, Z) -> Riemannian_grad(Q, alpha, Z)
Retr = (Q, H) -> exponentialRetraction(Q, H)
Qref = Matrix{Float64}(I, n, n)

# Optimization options
opts = Dict(
    :alpha_bar => 1.0,  #initial step size for line search
    :c         => 0.01, # parameter for Armijo condition
    :beta      => 0.7,  # step size reduction factor for line search
    :maxIter   => maxIter,
    :C         => C,    # radius for the universal PEL
    :tol       => 1e-9, # tolerance for convergence
    :alpha     => alpha
)

# define results storage
Qopts = Vector{Matrix{Float64}}(undef, length(Qs))
energies = zeros(length(Qs))
iterations = zeros(Int, length(Qs))

#------------------ Run optimization for each initial Q_0 ----------------------
total = length(Qs)
progress = Atomic{Int}(0)

@floop ThreadedEx() for i in eachindex(Qs)

    Q0 = Qs[i]

    Qopt, E_Q_arr, _, iter =  AcceleratedLineSearch_P(
        Q0,
        E,
        grad_E,
        Retr,
        opts,
        metric_type,
        Z_universal,
        evaluationUpdate
    )

    Qopts[i] = Qopt
    energies[i] = E_Q_arr[iter]
    iterations[i] = iter

    done = atomic_add!(progress, 1) + 1
    println("Completed $done / $total (thread $(threadid()))")
end

# Normalize and reduce final Qopts
Qopts = normalize_and_reduce.(Qopts)

#------------------ only keep Qs with iterations < maxIter-----------------------
keep_idx = findall(iterations .< maxIter)
Qopts    = Qopts[keep_idx]
energies = energies[keep_idx]
println("Kept $(length(Qopts)) entries after removing minimizers with max iterations")

#----------------------only keep unique Qopts------------------------------------
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

@floop ThreadedEx() for i in eachindex(Qopts)
    result, λ_min,_ = is_second_order_critical(Qopts[i], alpha, Z_universal)
    is_local_min[i] = result === true || result == true
    println("$(i) of $(eachindex(Qopts)) tested for local minimizer with λ_min=",round(λ_min, sigdigits=6))
end
Qopts      = Qopts[is_local_min]
energies   = energies[is_local_min]

println("Kept $(length(Qopts)) / $(length_Qopts_old) potential local minimizers.")


# ---------------------Save results to JLD2 and text file--------------------------------------

@save joinpath(results_dir,"results_$(Method)alphaPI","n_$(n)" , "$(Method)optimized_Qs_n$(n)_alphaPI_test.jld2") Qopts energies
save_matrices_txt(joinpath(results_dir,"results_$(Method)alphaPI","n_$(n)" , "$(Method)optimized_Qs_n$(n)_alphaPI_test.txt"), Qopts)