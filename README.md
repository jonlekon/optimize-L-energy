# optimize-L-energy
Develops Smooth Optimizations method to minimize the L-energy functional over the SPD Gram matrices with unit determinate.
#  Gaussian Core Model Minimization

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

The cost function we aim to minimize is

$$
\mathcal{E}_{\alpha}(Q) = \sum_{0 \ne z \in \mathbb Z^n} \mathrm{e}^{\alpha z^TQz}.
$$


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

  ---


##  Folder Structure and Workflow

This repository is organized into four main components:

* `experiments/` – scripts to run optimization algorithms and post-processing
* `input_data/` – required input data (initial Gram matrices and PELs)
* `results/` – output of optimization runs and analysis
* `src/` – core implementation of the algorithms

---

##  Running Optimization Experiments

All optimization scripts are located in the `experiments/` folder.

### 1. Prerequisites

Before running any optimization (`Run_...` scripts), you need:

* **Initial Gram matrices**
  Located in:
  `input_data/Initial_Gram_matrices/`

* **Universal Partial Evaluation Lattice (PEL)**
  Located in:
  `input_data/Partial_Evaluation_Lattices/`

These inputs are required for all optimization routines.

---

### 2. Running an Optimization

To run an optimization, execute one of the following scripts:

```bash
julia Run_*.jl
```

Examples:

* `Run_RiemannianTrustRegionMethod_universalPEL.jl`
* `Run_Riemannian_Gradient_Descent.jl`
* `Run_Riemannian_Newton_BasisMethod.jl`

Each script:

* loads the required input data,
* performs the optimization,
* stores the resulting Gram matrices in the `results/` folder.

---

### 3. Post-processing Results

After running an optimization, the resulting Gram matrices must be:

* beautified, and
* grouped into equivalence classes

This is done using:

```bash
julia CleanGramMatrices.jl
```

This script processes the raw output and stores cleaned and classified results.

---

### 4. Interpreting Results

To interpret the optimized Gram matrices, run:

```bash
julia "interpret optimal matrices.jl"
```

This provides a more meaningful representation of the computed solutions.

---

### 5. Optimization over Multiple α Values

To run optimizations over multiple values of α:

1. Execute one of:

   * `Run_OptimizeOverMultibleAlphas_n2.jl`
   * `Run_OptimizeOverMultibleAlphas_n3.jl`

2. Then generate plots using:

   * `plot_landscape.jl`
   * `plot_landscape_n2.jl`

---

### 6. Generating Input Data

The `experiments/` folder also contains scripts for generating input data.

* File names are self-explanatory
* These scripts generate:

  * initial Gram matrices
  * partial evaluation lattices

---

##  Folder Overview

```
.
├── experiments/        # optimization scripts and analysis tools
├── input_data/         # required input (Gram matrices, PELs)
├── results/            # outputs, plots, and processed data
└── src/                # core implementation
```

---

##  Notes

* Ensure that all required input data is available before running any `Run_...` script.
* Post-processing with `CleanGramMatrices.jl` is required for meaningful results.
* For multi-α experiments, plotting must be performed separately after optimization.

  
