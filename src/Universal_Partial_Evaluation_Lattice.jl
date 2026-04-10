using LinearAlgebra
using LLLplus
using Random
using JLD2
using DelimitedFiles
using Base.Threads

include("EvaluationMatrix.jl")
include("GeometricObjects.jl")
using .EvaluationMatrix: enumerate_quadint
using .GeometricObjects: exponentialRetraction

function build_PEM(Qs::Vector{<:AbstractMatrix}, C::Real, N::Int)
    
    N = min(N, length(Qs))
    n = size(Qs[1], 1)
    
    PEM = Matrix{Int}(undef, 0, n)   # empty matrix with n columns
   
     Threads.@threads for i in 1:N
        
        Q₀ = Qs[i]
        
        Z = enumerate_quadint(Q₀, C)   # k × n matrix
        
        if Z !== nothing && !isempty(Z)
            PEM = vcat(PEM, Z)         # merge
            PEM = unique(PEM; dims=1)  # remove duplicate rows
        end
       # println("Current Loop is: ", i)
    end
    
    return PEM
end

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

function tangent_projection_basis(Q)
   n = size(Q,1)
   basis = tangent_basis(Q)
   ξ = zeros(n,n)
   for i in size(basis,1)
    a_i = rand()
    ξ = ξ + a_i* basis[i]
   end
   return ξ
end

function random_matrix_Pn(n)
    eigvals = rand(n)
    #eigvals = [exp(vals) for vals in eigvals]
    #display(eigvals)
    Q = randn(n,n)
    Q = (Q + Q')/2
    Q,_ = qr(Q)
    Q = Q * Diagonal(eigvals)*Q'
    
    ξ = tangent_projection_basis(Q)
    Q = exponentialRetraction(Q,ξ)
    Q /= abs(det(Q))^(1/n)
end

function build_PEM_random(min_PEMgrowth::Real, C::Real, n::Int, maxNum::Int)
 
    PEM = Matrix{Int}(undef, 0, n)   # empty matrix with n columns
     Q = random_matrix_Pn(n)
     Z = enumerate_quadint(Q,C)
     PEM = vcat(PEM, Z) 
    current_PEMgrowth = size(PEM, 1) #number of inital vector added to PEM

     while current_PEMgrowth >= min_PEMgrowth && size(PEM, 1)<= maxNum
        PEM_size = size(PEM, 1)
        Q = random_matrix_Pn(n)  
        Z = enumerate_quadint(Q, C)   # k × n matrix
        
        if Z !== nothing && !isempty(Z)
            PEM = vcat(PEM, Z)         # merge
            PEM = unique(PEM; dims=1)  # remove duplicate rows
        end
        current_PEMgrowth = size(PEM, 1) - PEM_size
       # println("Current PEM growth is: ", current_PEMgrowth)
       # println("number of PEL vector:", size(PEM, 1))
    end
    
    return PEM
end
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

input_data = joinpath(@__DIR__, "..", "input_data")
n = 2
alpha = float(pi)
Method = "RTR_"
folder_name = joinpath(input_data, "Initial_Gram_matrices")

@load joinpath(folder_name,"Random_initial_Qs_n$(n).jld2") Qs
N = length(Qs)
C = radius_C(n)
println("Using radius C = ", C)
min_PEMgrowth = 0
maxNum = 2000  #maximum number of vectors in the PEM

PEM_random = build_PEM_random(min_PEMgrowth, C, n, maxNum)
PEM_initial = build_PEM(Qs, C, N)
PEM = vcat(PEM_random, PEM_initial)        
PEM = unique(PEM; dims=1)
println("Number of PEL vector: ", size(PEM, 1))
@save joinpath(input_data, "Partial_Evaluation_Lattices/", "universal_PEL_n$(n)_C$(C)_test.jld2") PEM
