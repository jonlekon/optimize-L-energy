# optimize-L-energy
Develops Smooth Optimizations method to minimize the L-energy functional over the SPD Gram matrices with unit determinate.
#  Gaussian Core Model Minimization on Lattices

This repository contains the implementation and experimental results for a master’s thesis on the **minimization of the Gaussian core model over lattices of fixed volume**.

##  Problem Setup

The problem is formulated over the space of lattices parametrized by their Gram matrices. Given a lattice basis matrix $B$, the associated Gram matrix is defined as

$$
Q = B^T B.
$$

Since lattices that differ only by scaling correspond to Gram matrices differing by a scalar multiple, we restrict the search to the set

$$
\mathcal{P}_n = \{ Q \in \mathrm{SPD}_n \mid \det(Q) = 1 \},
$$

i.e., the manifold of symmetric positive definite matrices with unit determinant. This constraint fixes the lattice co-volume, enabling meaningful comparisons between configurations.

The set $\mathcal{P}_n$ forms a smooth Riemannian manifold embedded in the space of symmetric matrices. This allows the use of methods from *Riemannian optimization*, which solve constrained problems by operating directly on the manifold using tools such as tangent spaces, retractions, and Riemannian gradients.

---

##  Methods

This project implements and compares several Riemannian optimization algorithms:

- Riemannian Gradient Descent  
- Riemannian Newton Method  
- Riemannian Trust-Region Method  

These methods are used to efficiently search for minimizers of the Gaussian core model under the manifold constraint.

---

##  Objectives

The project has two main goals:

1. **Theoretical framework**  
   Develop the differential geometric tools required for optimization on the manifold of determinant-one positive definite matrices.

2. **Numerical implementation**  
   Compute minimizers of the Gaussian core model:
   - For dimensions $n \in \{2,3,4,5,6,7,8\}$ at fixed parameter $\alpha$
   - For multiple values of $\alpha$ in dimension $n = 3$

---

##  Results

The repository includes:

- Numerical experiments across multiple dimensions  
- Comparisons between optimization methods  
- Visualizations and analysis of resulting lattice structures  
