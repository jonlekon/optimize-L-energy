using LinearAlgebra
using LLLplus


# Gram matrix from basis
gram(B) = B' * B

# orthonormalize basis (key fix)
function orthonormal_basis(B)
    Q, R = qr(B)
    return Matrix(Q)  # orthonormal frame
end


# compare invariant spectra
function gram_invariants(Q)
    ev = eigen(Symmetric(Q)).values
    sort!(ev)
    return ev
end

function are_congruent_robust(Q1, Q2; tol=1e-8)
    v1 = gram_invariants(Q1)
    v2 = gram_invariants(Q2)
    println("Eigenvalues of Q1: ", v1)
    println("Eigenvalues of Q2: ", v2)  

    return norm(v1 - v2, Inf) < tol
end


Q1 = [
    1.2599210498924531 0.6299605249456222 0.629960524944058;
0.6299605249456222 1.2599210498923075 0.6299605249413485;
0.629960524944058 0.6299605249413485 1.2599210498923383
]

Q2 = [
    1.2599210498860338  0.6299605249424548 -0.6299605249418;
    0.6299605249424548  1.2599210498995734 -0.6299605249581633;
   -0.6299605249418    -0.6299605249581633  1.2599210498990845
]

B1 = Matrix(cholesky(Q1).U)
        B1 = lll(B1)[1]
B2 = Matrix(cholesky(Q2).U)
        B2 = lll(B2)[1]  
display(B1)
display(B2)


Q1 = gram(B1)
Q2 = gram(B2)

println(are_congruent_robust(Q1, Q2))

