#works
using LinearAlgebra
using LLLplus
using Random
using JLD2
using DelimitedFiles
using Base.Threads


#function that samples random Q \in P = {Q \in S^n_++: det Q = 1}
function sample_initial_Qs_random(n::Int, N::Int, method::String="gaussian", useLLL::Bool=false)
   Qs = Vector{Matrix{Float64}}()

     for i = 1 : N
        if lowercase(method) == "gaussian"
            B = randn(n, n)
        elseif lowercase(method) == "orthogonal"
            U, _, V = svd(randn(n, n))
            B = U * V'
        else
            error("Unknown method. Use \"gaussian\" or \"orthogonal\".")
        end

        # Normalize covolume to 1
        B ./= abs(det(B))^(1/n)
        # Optionally apply LLL reduction
        if useLLL
            # Requires an external package (e.g. Nemo.jl or LLLplus.jl)
            # Apply LLL reduction
            B,_ = lll(B)
        end
        Q = B'*B
       if(i%100 == 0)
           # println("det is :", det(Q))
           # println(" is positive def :", isposdef(Q))
       end
        push!(Qs, Q)

   end
   return Qs

    return B
end

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

n = 2     # dimension
N = 10   # number of random initial Qs to sample
Qs = sample_initial_Qs_random(n, N, "gaussian", true)

# Save Gram matrices to text and JLD2 files
input_data = input_data = joinpath(@__DIR__, "..", "input_data")
folder_name = joinpath(input_data, "Initial_Gram_matrices/")
@save joinpath(folder_name, "Random_initial_Qs_n$(n)_test.jld2") Qs
save_matrices_txt(joinpath(folder_name, "Random_initial_Qs_n$(n)_test.jld2.txt"), Qs)






