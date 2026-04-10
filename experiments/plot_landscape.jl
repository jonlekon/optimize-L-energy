#works
using JLD2
using Plots
#using Revise

src = joinpath(@__DIR__, "..", "src")
results_dir= joinpath(@__DIR__, "..", "results")
input_data = joinpath(@__DIR__, "..", "input_data")

include(joinpath(src, "lattice_isometry.jl"))

#------------------------------Analysis of results---------------------------------------------
folder_name = "results\\alpha_critical_analysis\\temporary_results\\"
@load joinpath(folder_name, "checkpoint_Qopts.jld2") Qopts 
@load joinpath(folder_name, "checkpoint_energies.jld2") energies
@load joinpath(folder_name, "checkpoint_alphas.jld2") alphas
@load joinpath(folder_name, "checkpoint_lambda.jld2") lambda

FCC = 2^(-2/3)*[2 -1 -1;
                  -1 2  1;
                  -1  1 2  ] 
FCC_dual = 2^(-4/3)*[3 1 1
                    1 3 -1
                    1 -1 3]  

#--------Helper functions for clasify_clusters-------
#group by alpha
function group_by_alpha(alphas)
    groups = Dict{Float64, Vector{Int}}()
    for i in eachindex(alphas)
        push!(get!(groups, alphas[i], Int[]), i)
    end
    return groups
end

# find clusters by isometry. Extract equivalence classes
function cluster_minimizers(Qs; tol=1e-2)
    clusters = Vector{Tuple{Int, Vector{Int}}}()

    for i in eachindex(Qs)
        assigned = false

        for k in eachindex(clusters)
            rep_idx = clusters[k][1]

            if lattice_isometry(Qs[i], Qs[rep_idx]; tol=tol)
                push!(clusters[k][2], i)
                assigned = true
                break
            end
        end

        if !assigned
            push!(clusters, (i, [i]))
        end
    end

    return clusters
end

# clsify in FCC, FCC_dual and other
function classify_clusters(reps; tol=1e-2)
    labels = Symbol[]

    for Q in reps
        if lattice_isometry(Q, FCC; tol=tol)
            push!(labels, :FCC)
        elseif lattice_isometry(Q, FCC_dual; tol=tol)
            push!(labels, :FCC_dual)
        else
            push!(labels, :other)
        end
    end

    return labels
end

# what alpha belongs to which cluster
function analyze_landscape(Qopts, energies, alphas, lambda; tol=1e-2)
    groups = group_by_alpha(alphas)

    results = Dict()

     for (alpha, idxs) in groups
        Qsub = Qopts[idxs]
        λsub = lambda[idxs, :]

        clusters = cluster_minimizers(Qsub; tol=tol)

        reps = [Qsub[c[1]] for c in clusters]
        sizes = [length(c[2]) for c in clusters]

        labels = classify_clusters(reps; tol=tol)

        # minimal eigenvalue per cluster
        min_eigs = Float64[]
        for c in clusters
            eigvals_cluster = minimum(λsub[c[2], :], dims=2)
            push!(min_eigs, minimum(eigvals_cluster))
        end

        results[alpha] = (
            num_clusters = length(clusters),
            representatives = reps,
            sizes = sizes,
            labels = labels,
            min_eigs = min_eigs
        )
    end

    return results
end

#print the results

function print_summary(results)
    println("\n===== LANDSCAPE SUMMARY =====\n")

    for alpha in sort(collect(keys(results)))
        data = results[alpha]

        println("alpha = $alpha")
        println("  #clusters = ", data.num_clusters)
        println("  labels    = ", data.labels)
        println("  sizes     = ", data.sizes)
        println("  min eigs  = ", round.(data.min_eigs; sigdigits=4))
        println()
    end
end



results_landscape_analysis = analyze_landscape(Qopts, energies, alphas, lambda; tol=1e-2)

#---------------------Save Data for plotting---------------
function prepare_plot_data(results)
    alphas_sorted = sort(collect(keys(results)))

    num_clusters = Int[]
    largest_basin = Float64[]
    second_basin = Float64[]
    min_eig_global = Float64[]
    fcc_fraction = Float64[]
    # optional: store full info
    sizes_all = Vector{Vector{Int}}()
    labels_all = Vector{Vector{Symbol}}()
    min_eigs_all = Vector{Vector{Float64}}()

    for alpha in alphas_sorted
        data = results[alpha]
        total = sum(data.sizes)
        fcc_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label in (:FCC, :FCC_dual)
        ; init=0)
        push!(num_clusters, data.num_clusters)

        # sort basin sizes descending
        sizes_sorted = sort(data.sizes, rev=true)

        push!(largest_basin, sizes_sorted[1])
        push!(second_basin, length(sizes_sorted) > 1 ? sizes_sorted[2] : 0)

        push!(min_eig_global, minimum(data.min_eigs))

        push!(sizes_all, data.sizes)
        push!(labels_all, data.labels)
        push!(min_eigs_all, data.min_eigs)
        push!(fcc_fraction, fcc_size / total)
    end


    return (
        alphas = alphas_sorted,
        num_clusters = num_clusters,
        largest_basin = largest_basin,
        second_basin = second_basin,
        min_eig_global = min_eig_global,
        sizes_all = sizes_all,
        labels_all = labels_all,
        min_eigs_all = min_eigs_all,
        fcc_fraction = fcc_fraction
    )
end

plot_data = prepare_plot_data(results_landscape_analysis)

@save joinpath(results_dir, "alpha_critical_analysis","temporary_results" ,"landscape_analysis.jld2") plot_data results_landscape_analysis

function compute_fractions(results)
    alphas_sorted = sort(collect(keys(results)))

    fcc = Float64[]
    dual = Float64[]
    other = Float64[]

    for α in alphas_sorted
        data = results[α]

        total = sum(data.sizes)

        fcc_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :FCC
        ; init=0)

        dual_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :FCC_dual
        ; init=0)

        other_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :other
        ; init=0)

        push!(fcc, total > 0 ? fcc_size / total : 0.0)
        push!(dual, total > 0 ? dual_size / total : 0.0)
        push!(other, total > 0 ? other_size / total : 0.0)
    end

    return alphas_sorted, fcc, dual, other
end

function compute_class_numbers(results)
    alphas_sorted = sort(collect(keys(results)))

    fcc = Float64[]
    dual = Float64[]
    other = Float64[]

    for α in alphas_sorted
        data = results[α]

        total = sum(data.sizes)

        fcc_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :FCC
        ; init=0)

        dual_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :FCC_dual
        ; init=0)

        other_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :other
        ; init=0)

        push!(fcc, total > 0 ? fcc_size  : 0.0)
        push!(dual, total > 0 ? dual_size  : 0.0)
        push!(other, total > 0 ? other_size  : 0.0)
    end

    return alphas_sorted, fcc, dual, other
end


_, fcc_fraction, dual_fraction, other_fraction = compute_fractions(results_landscape_analysis)
_, fcc_numbers, dual_numbers, other_numbers = compute_class_numbers(results_landscape_analysis)

n = 3
α = plot_data.alphas
fig_dir = joinpath(results_dir, "alpha_critical_analysis", "Figures")

# Plot 1
ymax = maximum(plot_data.num_clusters)
xmin = minimum(α)
xmax = maximum(α)

p1 = plot(α, plot_data.num_clusters,
    xlabel="alpha",
    ylabel="Number of potential optimal lattices",
    marker=:o,
    xlim=((xmin-0.1),5.1),
    ylim=(0, ymax + 1),
    xticks = xmin:0.2:xmax,
    label = "Number of potential optimal lattices"
)
savefig(p1, joinpath(fig_dir, "num_minimizers.pdf"))


# Plot 2
p2 = plot(α, plot_data.largest_basin,
    label="Largest basin",
    xlabel="alpha",
    ylabel="Basin size",
    xlim =(xmin, xmax),
    xticks = xmin:0.5:xmax,
    marker=:o)

plot!(p2, α, plot_data.second_basin,
    label="Second largest basin",
    marker=:o)

savefig(p2, joinpath(fig_dir, "basins.pdf"))


p3 = plot(α, plot_data.min_eig_global,
    xlabel="alpha",
    ylabel="Smallest Eigenvalue of the Hessian matrix",
    marker=:o,
    xticks = xmin:(xmax/10):xmax,
     xlim=(xmin, xmax),
    label="Stability of minimizers"
)

hline!(p3, [0], color=:red, linestyle=:dash, label="second-order criticality threshold")

savefig(p3, joinpath(fig_dir, "stability.pdf"))

# Plot 4
p4 = plot(α, fcc_fraction,
    xlabel="alpha",
    ylabel="Fraction of runs",
    marker=:o,
    title="Fraction of runs converging to FCC-type minimizers",
    ylim=(0, 1.2),
    label="FCC + FCC_dual",
    xlim=(xmin, xmax),
    xticks = xmin:0.2:xmax)

savefig(p4, joinpath(fig_dir, "fcc_fraction.pdf") )


# Plot 5
p = plot(α, fcc_fraction,
    label="FCC",
    xlabel="alpha",
    ylabel="Fraction of runs",
    marker=:o,
    ylim=(0,1.1),
    xlim=(xmin, xmax),
    xticks = xmin:0.5:xmax,
    linewidth=2
)

plot!(p, α, dual_fraction,
    label="FCC dual",
    marker=:o,
    linewidth=2
)

plot!(p, α, other_fraction,
    label="Other",
    marker=:o,
    linewidth=2
)

savefig(p, joinpath(fig_dir, "fraction_breakdown.pdf"))



# Plot 6: number of classes per type
p = plot(α, fcc_numbers,
    label="FCC",
    xlabel="alpha",
    ylabel="Total cluster frequencies",
    marker=:o,
    ylim=(-0.1, maximum(dual_numbers) + 10),
     xlim=((xmin-0.1),5.1),
    linewidth=2,
    xticks = xmin:0.5:xmax,
    legend = :top 
)

plot!(p, α, dual_numbers,
    label="dual FCC",
    marker=:o,
    linewidth=2
)

plot!(p, α, other_numbers,
    label="Other",
    marker=:o,
    linewidth=2
)

savefig(p, joinpath(fig_dir, "number_breakdown.pdf"))   