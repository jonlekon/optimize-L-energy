module GeometricObjects

using LinearAlgebra
using Base.Threads
using LoopVectorization
using JLD2
using DelimitedFiles
#using Revise

export compute_E_energy
export compute_E_energy_fast
export Riemannian_grad
export exponentialRetraction
export Riemannian_Hessian

"""
Module: GeometricObjects

This module provides core geometric and differential operators for the analysis
and optimization of lattice energy functionals on the manifold

    𝒫ₙ = { Q ∈ Sym⁺(n) : det(Q) = 1 },

the space of symmetric positive definite matrices with unit determinant,
equipped with the affine-invariant Riemannian metric.

The module implements the following key components:

1. Energy Functional
--------------------
- `compute_E_energy(Q, alpha, Zmat)`:
  Evaluates the lattice energy
      E_alpha(Q) = ∑_{z ∈ Z} exp(-alpha zᵀ Q z),
  where `Zmat` contains evaluation vectors z ∈ ℤⁿ excluding 0
  This corresponds to a Gaussian-type interaction energy.

2. First-Order Geometry
----------------------
- `Riemannian_grad(Q, alpha, Zmat)`:
  Computes the Riemannian gradient of the energy functional under the
  affine-invariant metric. The result is projected onto the tangent space
  of 𝒫ₙ, ensuring the unit determinant constraint is respected.

- `tangentProjection(X, Q)`:
  Projects a symmetric matrix onto the tangent space at Q by removing the
  radial (trace) component. This enforces the constraint det(Q) = 1.

3. Retractions / Manifold Updates
--------------------------------
- `exponentialRetraction(Q, H)`:
  Implements the exponential map on the SPD manifold using spectral
  decomposition. This provides a geometrically exact update along a tangent
  direction H.

- `exponentialRetraction_2(Q, H; alpha)`:
  A scaled variant of the exponential retraction, useful for step-size control
  (e.g., in line search or trust-region methods).

4. Second-Order Geometry
-----------------------
- `Riemannian_Hessian(Q, H, alpha, Zmat)`:
  Computes the Riemannian Hessian applied to a tangent vector H.
  The result is projected back onto the tangent space, yielding the
  second-order variation of the energy functional.

Applications
------------
This module is designed for:
- studying the energy landscape of lattice energies,
- detecting local and global minimizers,
- analyzing stability via Hessian eigenvalues,
- investigating bifurcation phenomena as parameters (e.g., alpha) vary.

The implementation emphasizes:
- vectorized computations for efficiency,
- geometric consistency with the affine-invariant metric,
- compatibility with Riemannian optimization algorithms.

"""


"""
   compute_L_energy(Q::AbstractMatrix{<:Real},
                             alpha::Real,
                             Zmat::AbstractMatrix{<:Real})

Compute the E-energy given a gram matrix `Q`, a parameter `alpha`, and a evaluation matrix `Zmat`.
"""
function compute_E_energy(Q::AbstractMatrix{<:Real},
                         alpha::Real,
                         Zmat::AbstractMatrix{<:Real})

    E = zero(eltype(Q))
    for i in axes(Zmat, 1)
        z = view(Zmat, i, :)
        q = dot(z, Q * z)
        E += exp(-alpha * q)
    end
    return E
end


"""
    Riemannian_grad(Q, alpha, Zmat)

Compute the Riemannian gradient

    grad E_alpha(Q) = -alpha · Q · (∑_z e^{-alpha zᵀQz} z zᵀ) · Q

where rows of `Zmat` are integer vectors z ∈ ℤⁿ ∖ {0}.

Arguments:
- `Q`    : n×n symmetric (SPD) matrix
- `alpha`    : positive scalar
- `Zmat` : k×n matrix, each row is a vector z

Returns:
- `G`    : n×n matrix, Riemannian gradient projected onto the tangent space
"""
function Riemannian_grad(
    Q::AbstractMatrix{<:Real},
    alpha::Real,
    Zmat::AbstractMatrix{<:Real}
)
    n = size(Q, 1)

    # -------------------- Q[z] = zᵀ Q z for all rows --------------------
    # Vector of length k
    Qz = sum((Zmat * Q) .* Zmat, dims=2)[:, 1]

    # -------------------- Weights --------------------
    w = exp.(-alpha .* Qz)   # elementwise exp

    # -------------------- Σ wᵢ zᵢ zᵢᵀ --------------------
    # Equivalent to Zmat' * diagm(w) * Zmat
    # but avoids forming diagm(w)
    Sz = Zmat' * (Zmat .* w)

    # -------------------- Riemannian gradient --------------------
    G = -alpha * Q * Sz * Q

    # Project onto tangent space at Q
    return tangentProjection(G, Q)
end

"""
    tangentProjection(X, Q)

Project matrix `X` onto the tangent space at `Q` of the SPD manifold
with the affine-invariant metric.

This removes the radial component proportional to `Q`.
"""
function tangentProjection(
    X::AbstractMatrix{<:Real},
    Q::AbstractMatrix{<:Real}
)
    n = size(Q, 1)

    # Cholesky factorization Q = R' * R
    R = cholesky(Q).U

    # Compute Q^{-1} * X using triangular solves
    QiX = R \ (R' \ X)

    # Remove trace component
    return X - (tr(QiX) / n) * Q
end


"""
    exponentialRetraction(Q, H)

Compute the exponential retraction on the manifold
P = {Q ∈ S_++^n : det(Q) = 1} under the affine-invariant metric.

Arguments:
- Q : n×n symmetric positive definite matrix (det(Q) = 1)
- H : n×n symmetric tangent vector at Q

Returns:
- Y : n×n symmetric positive definite matrix on the manifold
"""
function exponentialRetraction(Q::AbstractMatrix{<:Real},
                               H::AbstractMatrix{<:Real})

    n = size(Q,1)

    # Eigen of Q
    F = eigen(Q)
    U = F.vectors
    d = max.(F.values, 1e-12)  # safety

    Qhalf  = U * Diagonal(sqrt.(d)) * U'
    Qihalf = U * Diagonal(1.0 ./ sqrt.(d)) * U'

    S = Qihalf * H * Qihalf
    S = 0.5 * (S + S')  # ensure symmetry
    if any(!isfinite, S)
        S = Matrix{Float64}(I, n, n)
    end
    E = exp(S)
    Y = Qhalf * E * Qhalf
    Y = 0.5 * (Y + Y')  # ensure symmetry
    
    # determinant normalization is not necesarry 
    return Matrix(Y)
end

function exponentialRetraction_2(Q, H; alpha=1.0)
    n = size(Q,1)
    
    F = eigen(Q)
    U = F.vectors
    d = F.values
    Qhalf  = U * Diagonal(sqrt.(d)) * U'
    Qihalf = U * Diagonal(1 ./ sqrt.(d)) * U'
    
    S = Qihalf * (alpha * H) * Qihalf
    S = 0.5 * (S + S')
    
    E = exp(S)
    Y = Qhalf * E * Qhalf
    Y = 0.5 * (Y + Y')
    
    return Y
end


"""
    Riemannian_Hessian(Q, H, alpha, Zmat)

Compute the Riemannian Hessian of the energy E_alpha(Q) at point Q
along tangent direction H, under the affine-invariant metric.

Arguments:
- Q    : n×n SPD matrix (point on the manifold)
- H    : n×n symmetric tangent vector at Q
- alpha    : scalar parameter
- Zmat : k×n matrix whose rows are z ∈ ℤ^n (evaluation lattice)

Returns:
- Hess : n×n matrix, Riemannian Hessian projected onto tangent space
"""
function Riemannian_Hessian(Q::AbstractMatrix{<:Real},
                            H::AbstractMatrix{<:Real},
                            alpha::Real,
                            Zmat::AbstractMatrix{<:Real})

    n, k = size(Q, 1), size(Zmat, 1)

    # ---------------- Quadratic forms ----------------
    Qz = vec(sum((Zmat * Q) .* Zmat, dims=2))  # Q[z] = z' Q z
    Hz = vec(sum((Zmat * H) .* Zmat, dims=2))  # H[z] = z' H z

    # ---------------- Weights ----------------
    w = exp.(-alpha .* Qz)                           # vector of length k

    # ---------------- Precompute ----------------
    ZQ = Zmat * Q                                # k × n

    # ---- First term: sum w_z * alpha * H[z] * Q z z' Q ----
    S1 = ZQ' * (ZQ .* (w .* Hz))                # n × n matrix

    # ---- Second term: sum w_z * 1/2 * (R_z + R_z') ----
    # R_z = H z z' Q, vectorized
    S2 = H * (Zmat' * (ZQ .* w))                # n × n
    S2 = 0.5 * (S2 + S2')                        # symmetrize

    # ---- Hessian ----
    Hess = alpha * (alpha * S1 - S2)

    # Project onto tangent space at Q
    Hess = tangentProjection(Hess, Q)

    return Hess
end

end # module
