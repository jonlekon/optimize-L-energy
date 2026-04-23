using LinearAlgebra
using LLLplus
using Random
using JLD2
using DelimitedFiles
using Plots
using Plots.PlotMeasures

src = joinpath(@__DIR__, "..", "src")
input_data = joinpath(@__DIR__, "..", "input_data")
results_dir = joinpath(@__DIR__, "..", "results")

include(joinpath(src, "GeometricObjects.jl"))
include(joinpath(src, "module second_order_critical.jl"))

using .GeometricObjects: compute_E_energy, Riemannian_Hessian, Riemannian_grad
using .second_order_critical: is_second_order_critical


n = 3
C= 300
Q_FCC_dual = 2^(-4/3)*[3 1 1
                      1 3 -1
                      1 -1 3]

Q_FCC = 2^(-2/3)*[2 -1 -1
                  -1 2 1
                  -1 1 2]

Q_T_1 = 2^(-4/3) *[4 2 2
                   2 3 1
                   2 1 3]
                 
folder_PEM = joinpath(input_data, "Partial_Evaluation_Lattices/")
@load joinpath(folder_PEM, "universal_PEL_n$(n)_C$(C).jld2") PEM
Zmat = PEM


# Function to compute E(alpha) for a given Q
function compute_E_values(alphas, Q, Zmat)
    return [compute_E_energy(Q, α, Zmat) for α in alphas]
end

function compute_lambdas(alphas, Q, Zmat)
        λmins = Float64[]
    
    for α in alphas
        _, λ_min, _ = is_second_order_critical(Q, α, Zmat)
        push!(λmins, λ_min)
    end
    
    return λmins
end

function compute_first_order(alphas, Q, Zmat)
        norm_grad = Float64[]
    
    for α in alphas
        G = Riemannian_grad(Q, α, Zmat)
        Q_inv = inv(Q)
        norm_G = sqrt(tr(Q_inv * G * Q_inv * G))
        push!(norm_grad, norm_G)
    end
    
    return norm_grad
end

# Main routine
# Main routine
function plot_E_vs_alpha(alphas, Q_FCC_dual, Q_T_1, Zmat, fig_dir)
    # Compute values
    E_FCC = compute_E_values(alphas, Q_FCC_dual, Zmat)
    E_T1  = compute_E_values(alphas, Q_T_1, Zmat)

    # Take logarithm
    logE_FCC = log.(E_FCC)
    logE_T1  = log.(E_T1)

    # Plot
    p1 = plot(alphas, logE_FCC,
              label="log(E) Q_FCC_dual",
              lw=2,
              linestyle=:solid)

    plot!(p1, alphas, logE_T1,
          label="log(E) Q_T_1",
          lw=2,
          linestyle=:dash)

    xlabel!(p1, "alpha")
    ylabel!(p1, "log(E(alpha))")
    title!(p1, "log(E(alpha)) vs alpha")

    # Save
    savefig(p1, joinpath(fig_dir, "log_E_alpha.pdf"))

    return p1
end

function plot_lambda_min_vs_alpha(alphas, Q_FCC,Q_FCC_dual, Q_T_1 ,Zmat, fig_dir)
    # Compute values
    lambdas_FCC = compute_lambdas(alphas, Q_FCC, Zmat)
    lambdas_dual_FCC = compute_lambdas(alphas, Q_FCC_dual, Zmat)
    lambdas_T_1  = compute_lambdas(alphas, Q_T_1, Zmat)


    # Plot
    p1 = plot(alphas, lambdas_FCC,
              label="lambda for FCC",
              lw=2,
              linestyle=:solid,
              size = (800, 300),  # (width, height) → reduced height
              left_margin = 10mm,
              bottom_margin = 8mm)

    plot!(p1, alphas, lambdas_dual_FCC,
          label="lambda for dual FCC",
          lw=2,
          linestyle=:solid)
    plot!(p1, alphas, lambdas_T_1,
    label="lambda for Q_T_1",
    lw=2,
    linestyle=:solid)

    xlabel!(p1, "alpha")
    ylabel!(p1, "min lambda")
    hline!(p1, [0], color=:red, linestyle=:dash, label="second-order criticality threshold")

    # Save
    savefig(p1, joinpath(fig_dir, "second_order_condition.pdf"))

    return p1
end


function plot_first_order_condition(alphas, Q_FCC,Q_FCC_dual, Q_T_1 ,Zmat, fig_dir)
    # Compute values
    norm_grads_FCC = compute_first_order(alphas, Q_FCC, Zmat)
    norm_grads_dual_FCC = compute_first_order(alphas, Q_FCC_dual, Zmat)
    norm_grads_T_1  = compute_first_order(alphas, Q_T_1, Zmat)


    # Plot
    p1 = plot(alphas, norm_grads_FCC,
              label="first order condtion for FCC",
              lw=2,
              linestyle=:solid,
              size = (800, 300),  # (width, height) → reduced height
              left_margin = 10mm,
              bottom_margin = 8mm,)

    plot!(p1, alphas, norm_grads_dual_FCC,
          label="first order condtion for dual FCC",
          lw=2,
          linestyle=:solid)
    plot!(p1, alphas, norm_grads_T_1,
    label="first order condtion for Q_T_1",
    lw=2,
    linestyle=:solid)

    xlabel!(p1, "alpha")
    ylabel!(p1, "norm of gradient")
    

    # Save
    savefig(p1, joinpath(fig_dir, "first_order_condition.pdf"))

    return p1
end

alphas = alphas = vcat(
    collect(0.4:0.1:10),     # very fine near transition
)
fig_dir = joinpath(results_dir, "alpha_critical_analysis", "Figures")

# make sure directory exists
isdir(fig_dir) || mkdir(fig_dir)

#p1 = plot_E_vs_alpha(alphas, Q_FCC_dual, Q_T_1, Zmat, fig_dir)
p2 = plot_lambda_min_vs_alpha(alphas,Q_FCC, Q_FCC_dual, Q_T_1, Zmat, fig_dir)
#p3 = plot_first_order_condition(alphas, Q_FCC,Q_FCC_dual, Q_T_1 ,Zmat, fig_dir)


