module CleanGramMatrices
#works
src = joinpath(@__DIR__, "..", "src")
input_data = joinpath(@__DIR__, "..", "input_data")
results = joinpath(@__DIR__, "..", "results")
include(joinpath(src, "module Detect_Closed_Expressions.jl"))
include(joinpath(src, "EvaluationMatrix.jl"))

using JLD2
using DelimitedFiles
using LinearAlgebra
using Statistics
using LLLplus
#using Revise

using .Detect_Closed_Expressions: detect_matrix 
using .EvaluationMatrix: shortest_lattice_vector, shortest_n_lattice_vectors

export lattice_isometry

  
n =  2   # dimension of the lattices 
alpha = float(pi) # parameter alpha in the energy functional
Method = "RTR_"

# -------------------- Load data of the optimizes Gram matrices --------------------
folder_name =joinpath(results,"results_$(Method)alphaPI","n_$(n)")
file_in = joinpath(folder_name,
                  "$(Method)optimized_Qs_n$(n)_alphaPI_test.jld2")

@assert isfile(file_in) "File not found: $file_in"
# Load data

data = load(file_in)
Qopts = data["Qopts"]
energies = data["energies"]

#------------------------Clean Matrices-------------------------------------------------------
# Optimization algorithms find matrices entries that are the same up to tol. clean_Matrices takes all matrices and
# compares similar values in magnitute with each other. It finds the right value in calculating the 
#average value and stores it at the affected entry
"""
    clean_Matrices(Qs::Vector{<:AbstractMatrix}, tol::Real=1e-6)

Cleans an array of matrices `Qs` by averaging entries that are numerically
equal up to `tol` (absolute value comparison). Preserves the original sign
of each entry.

# Arguments
- `Qs`: Array of matrices to clean.
- `tol`: Tolerance for considering entries equivalent (default `1e-6`).

# Returns
- `Qs_clean`: Array of cleaned matrices (same size as input).
"""
function clean_Matrices(Qs::Vector{<:AbstractMatrix}, tol::Real=1e-6)
    # Copy matrices to avoid mutating input
    Qs_clean = [copy(Q) for Q in Qs]
    
    # Track which entries were already updated
    updated = [falses(size(Q)) for Q in Qs_clean]

    # Helper: check if two numbers are approx equal in absolute value
    approx_equal(x, y) = abs(abs(x) - abs(y)) <= tol

    # Loop over all matrices and all entries
    for m1 in 1:length(Qs_clean)
        Q1 = Qs_clean[m1]
        n, _ = size(Q1)
        
        for i in 1:n
            for j in 1:n
                # Skip if already updated
                if updated[m1][i,j]
                    continue
                end

                x1 = Q1[i,j]

                # List to store all entries approx equal to x1
                indices = [(m1, i, j)]
                abs_vals = [abs(x1)]
                signs = [sign(x1)]

                # Compare with all entries in all matrices
                for m2 in 1:length(Qs_clean)
                    Q2 = Qs_clean[m2]
                    n2, _ = size(Q2)
                    for r in 1:n2
                        for c in 1:n2
                            # Skip current entry
                            if m1 == m2 && r == i && c == j
                                continue
                            end

                            x2 = Q2[r,c]
                            if approx_equal(x1, x2)
                                push!(indices, (m2, r, c))
                                push!(abs_vals, abs(x2))
                                push!(signs, sign(x2))
                            end
                        end
                    end
                end

                # Compute moving average of absolute values
                avg_abs = mean(abs_vals)

                # Apply avg_abs with the original signs to all matched entries
                for k in 1:length(indices)
                    m, r, c = indices[k]
                    Qs_clean[m][r,c] = avg_abs * signs[k]
                    updated[m][r,c] = true
                end
            end
        end
        Q1 = Q1/BigFloat(det(Q1)^(1/n))
    end

    return Qs_clean
end
Qopts_clean = clean_Matrices(Qopts, 1e-8) 


#----------------------find all unique equivalence classes----------------------------------

       #---------------------Look for Congurence to get only unique Grams and Lattice Bases-------
       #See which matrices generate different lattices
       # They define the same lattice iff there exists U in GL_n(Z) such that
       # we have Congruence transformation Q2 = U' Q1 U 

function sqrtm_sym(Q)
F = eigen(Symmetric(Q))
return F.vectors * Diagonal(sqrt.(F.values))
end 

function congurence_difference(Q1::AbstractMatrix, Q2::AbstractMatrix)
    #want to see if Q2 = U'* Q1 * U for U = B1^-1 * B2 in GL(Z)
    R1 = sqrtm_sym(Q1)
    R2 = sqrtm_sym(Q2)
    B1 = lll(R1)[1]
    B2 = lll(R2)[1]
    Q1 = transpose(B1) * B1
    Q2 = transpose(B2) * B2
    return abs(tr(Q1-Q2))  
end

#-----------------------Test for isometry of lattices------------------------------------------------------
        # Two lattices are suspected to be the same if they have  congruent Gram matrices or have  the same shortest lattice vectors
        # Beautification of the results
function lattice_isometry(Q1::AbstractMatrix, Q2::AbstractMatrix; tol=1e-2)

    #1 shortest lattice Vector
    _,SLV1 = shortest_lattice_vector(Q1)
    _,SLV2 = shortest_lattice_vector(Q2)
    diff_shortest_lattice_vectors = abs(SLV1-SLV2)# sqrt(norm(SLVs1-SLVs2))
    
    #2 congurence relation
    diff_congurence = congurence_difference(Q1,Q2)/ sqrt(n)
    #println("Congurence difference: ", diff_congurence)
    
    min_iso_diff = min(diff_congurence,  diff_shortest_lattice_vectors)
    println("iso difference: ",min_iso_diff)
    return min_iso_diff < tol
end


equiv_classes = Vector{Vector{Matrix{Float64}}}()

for Q in Qopts_clean
    placed = false
    for cls in equiv_classes
        if lattice_isometry(Q, cls[1])
            push!(cls, Q)
            placed = true
            break
        end
    end

    if !placed
        push!(equiv_classes, [Q])
    end
end

println("Number of equivalence classes: $(length(equiv_classes))")
@save joinpath(folder_name,"$(Method)optimized_Qs_n$(n)_alphaPI_equiv_classes_test.jld2") equiv_classes


#----------------------find closed expressions------------------------------------
# optimal lattices are suspected to have closed expressions, i.e. fractions with rational exponents
closed_equi_classes = [(detect_matrix(cls[1], tol=1e-8)) for cls in equiv_classes]   

#----------------------txt.file functions-------------------------------------
function save_equiv_classes_txt_Q(filename::AbstractString, closed_equi_classes)
    open(filename, "w") do io
        for (i, cls) in enumerate(closed_equi_classes)

            Qsym = cls.symbolic          # symbolic matrix

            println(io, "# Equivalence Class $i")
            println(io, "## Representative Gram matrix Q (symbolic)")

            for r in 1:size(Qsym, 1)
                println(io, join(string.(Qsym[r, :]), "  "))
            end

            println(io)
            println(io, "#"^60)
            println(io)
        end
        println(io,"Distribution of classes: $(map(length, equiv_classes))")
    end

    #println("Saved $(length(closed_equi_classes)) equivalence classes to $filename")
end

save_equiv_classes_txt_Q(joinpath(folder_name, "$(Method)optimized_Qs_n$(n)_alphaPI_equiv_classes_test.txt"), closed_equi_classes)

end # module CleanGramMatrices
