module EvaluationMatrix

export enumerate_quadint, shortest_lattice_vector, shortest_n_lattice_vectors

using LinearAlgebra
using LLLplus
using JLD2
using Base.Threads

"""
    enumerate_quadint(A, C)

Enumerate integer solutions x ≠ 0 of x' * A * x ≤ C
using Cholesky, unimodular LLL, permutation, Fincke-Pohst.

Returns a matrix X where each row is a solution vector x.
"""
function enumerate_quadint(A::AbstractMatrix{<:Real}, C::Real)

    # -------------------- Input checks --------------------
    n1, n2 = size(A)
    n1 == n2 || error("A must be square")
    n = n1
    issymmetric(A) || error("A must be symmetric")
    C > 0 || error("C must be positive")

    # -------------------- Step 1: Cholesky --------------------
    # Step 1: Cholesky
    # A = R' * R
      R = Matrix(cholesky(A).U)
    # -------------------- Step 2: LLL reduction --------------------
    # Bred = R * U, U unimodular integer
       Bred, U, _, _ = lll(R)
       U = round.(Int, U)
       S = Bred
       Sinv = U \ (R \ I(n))
    # -------------------- Step 3: permutation --------------------
    SinvT = Sinv'
    colnorms = vec(norm.(eachcol(SinvT)))
    pi_perm = sortperm(colnorms, rev=true)  # descending
    # apply permutation
    S = S[:, pi_perm]
    # inverse permutation
    invp = zeros(Int, n)
    for i in 1:n
        invp[pi_perm[i]] = i
    end

    # -------------------- Step 4: Q = S' * S --------------------
        Q = S' * S
        Rq = cholesky(Q).U
    # -------------------- Step 5: Fincke-ohst enumeration --------------------
    # Y is m × n matrix where each row is a solution vector y
    #Y =  fincke_pohst(Rq, C, n)
    Y = schnorr_euchner(Rq, C)
    m = size(Y, 1)
    Xrows = zeros(Int, 0, n)

    for r in 1:m
        y_perm = Y[r, :]  # row vector
        # reconstruct y_orig in original ordering
        y_orig = zeros(Int, n)
        for i in 1:n
            y_orig[pi_perm[i]] = round(Int, y_perm[i])
        end
        x = round.(Int, U * y_orig)
        if any(x .!= 0) && (x' * A * x <= C + 1e-8)
            Xrows = vcat(Xrows, x')  # append as row
        end
    end

    if isempty(Xrows)
        return zeros(Int, 1, n)
    else
        return unique(Xrows; dims=1)
    end
end

"""
    fincke_pohst(Rq, C, n)

Fincke–Pohst enumeration (recursive, robust).

Inputs:
- Rq : n×n upper-triangular matrix (Cholesky factor of Q)
- C  : nonnegative scalar bound
- n  : dimension

Output:
- Y  : m×n matrix whose rows are integer vectors y ≠ 0
       satisfying y' * Q * y ≤ C
"""
function fincke_pohst(Rq::AbstractMatrix{<:Real}, C::Real, n::Int)

    size(Rq,1) == size(Rq,2) || error("Rq must be square")
    C ≥ 0 || return zeros(Int, 0, n)

    d = diag(Rq)
    y = zeros(Float64, n)
    results = Vector{Vector{Int}}()

    function dfs(k::Int, rem::Float64) # deapth first search:
        #start at a “root node”
        #explore one branch as deep as possible
        # backtrack when you reach a leaf or dead end
        # repeat for the next branch

        rem < -1e-12 && return

        if k == 0
            if any(y .!= 0)
                v = Rq * y
                dot(v, v) ≤ C + 1e-10 && push!(results, round.(Int, y))
            end
            return
        end

        center = k == n ? 0.0 :
            -dot(Rq[k, k+1:end], y[k+1:end]) / Rq[k, k]

        radius = sqrt(max(rem, 0.0))
        lb = ceil(Int, center - radius / d[k])
        ub = floor(Int, center + radius / d[k])

        Threads.@threads for tk in lb:ub
            y[k] = tk
            delta = (d[k] * (y[k] - center))^2
            dfs(k - 1, rem - delta)
        end

        y[k] = 0.0
    end

    dfs(n, float(C))

    isempty(results) && return zeros(Int, 0, n)
    return unique(reduce(vcat, (r' for r in results)); dims=1)
end

#schnorr euchner enumeration can be used instead of fincke pohst

function schnorr_euchner(Rq::AbstractMatrix{<:Real}, C::Real)
    n = size(Rq, 1)
    size(Rq, 1) == size(Rq, 2) || error("Rq must be square")
    C ≥ 0 || return zeros(Int, 0, n)

    d = diag(Rq)                  # diagonal of R
    y = zeros(Float64, n)         # current vector being built
    results = Vector{Vector{Int}}()  # store valid integer vectors

    # Generate enumeration order closest to center
    function se_order(center::Float64, lb::Int, ub::Int)
        # start with integer closest to center
        mid = round(Int, center)
        order = Int[]
        push!(order, clamp(mid, lb, ub))

        offset = 1
        while true
            added = false
            for val in (mid + offset, mid - offset)
                if lb ≤ val ≤ ub
                    push!(order, val)
                    added = true
                end
            end
            if !added # no more integers in bounds
                break
            end
            offset += 1
        end
        return order
    end

    # Recursive DFS
    function dfs(k::Int, rem::Float64)
        rem < -1e-12 && return

        if k == 0
            if any(y .!= 0)
                v = Rq * y
                dot(v, v) ≤ C + 1e-10 && push!(results, round.(Int, y))
            end
            return
        end

        center = k == n ? 0.0 : -dot(Rq[k, k+1:end], y[k+1:end]) / Rq[k, k]
        radius = sqrt(max(rem, 0.0))

        # bounds for feasible integers
        lb = ceil(Int, center - radius / d[k])
        ub = floor(Int, center + radius / d[k])

        # SE enumeration order
        for tk in se_order(center, lb, ub)
            y[k] = tk
            delta = (d[k] * (y[k] - center))^2
            dfs(k - 1, rem - delta)
        end

        y[k] = 0.0  # reset for backtracking
    end

    dfs(n, float(C))

    isempty(results) && return zeros(Int, 0, n)
    return unique(reduce(vcat, (r' for r in results)); dims=1)
end


function lexsort_rows(M::AbstractMatrix)
    # M = m x n
    inds = sortperm(1:size(M,1), by = i -> M[i, :])
    return M[inds, :]
end


function shortest_lattice_vector(Q::AbstractMatrix)
    #function that return the norm of the shortest vector of the lattice induced by the gram matrix Q \in P
    Y = enumerate_quadint(Q, 5)
    @assert size(Y, 1) > 0 "Y must contain at least one vector"

    values = [
        view(Y, i, :)' * Q * view(Y, i, :)
        for i in 1:size(Y, 1)
    ]

    idx = argmin(values)

    return view(Y, idx, :), values[idx]
end

function shortest_n_lattice_vectors(Q::AbstractMatrix)
    #function that return the norm of the shortest vector of the lattice induced by the gram matrix Q \in P
    n = size(Q,1)
    Y = enumerate_quadint(Q, 5)
    @assert size(Y, 1) >= n "Y must contain at least n vectors"
   k = size(Y, 1)
    squared_lengths = [
        view(Y, i, :)' * Q * view(Y, i, :)
        for i in 1:k
    ]
    lengths = [sqrt(squared_lengths[i]) for i in 1:k]
    lengths = unique(lengths)
    lengths = sort(lengths)
    #println(lengths)

    return lengths[1:n]
end

end # module EvaluationMatrix