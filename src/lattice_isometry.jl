using LinearAlgebra
using LLLplus
using Random
using JLD2
using DelimitedFiles
using Base.Threads
#using Revise

src = joinpath(@__DIR__, "..", "src")
include(joinpath(src, "EvaluationMatrix.jl"))

using .EvaluationMatrix: shortest_lattice_vector

export lattice_isometry

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
    n = size(Q1, 1)
    #1 shortest lattice Vector
    _,SLV1 = shortest_lattice_vector(Q1)
    _,SLV2 = shortest_lattice_vector(Q2)
    diff_shortest_lattice_vectors = abs(SLV1-SLV2)# sqrt(norm(SLVs1-SLVs2))
    
    #2 congurence relation
    diff_congurence = congurence_difference(Q1,Q2)/ sqrt(n)
    #println("Congurence difference: ", diff_congurence)
    
    min_iso_diff = min(diff_congurence,  diff_shortest_lattice_vectors)
    #println("iso difference: ",min_iso_diff)
    return min_iso_diff < tol
end