using JLD2
using Plots
using Plots.PlotMeasures
#using Revise

src = joinpath(@__DIR__, "..", "src")
results_dir= joinpath(@__DIR__, "..", "results")
input_data = joinpath(@__DIR__, "..", "input_data")
experiments_dir = joinpath(@__DIR__, "..", "experiments")

include(joinpath(src, "lattice_isometry.jl"))

#------------------------------Analysis of results---------------------------------------------
folder_name = "results\\alpha_critical_analysis\\alphas_n2_01_10\\"
@load joinpath(folder_name, "checkpoint_Qopts.jld2") Qopts 
@load joinpath(folder_name, "checkpoint_energies.jld2") energies
@load joinpath(folder_name, "checkpoint_alphas.jld2") alphas
@load joinpath(folder_name, "checkpoint_lambda.jld2") lambda

Q_hex  = (1 / sqrt(3)) *[ 2  1;
                             1  2]
 
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
function cluster_minimizers(Qs; tol=1e-1)
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

# clsify in HEX and other
function classify_clusters(reps; tol=1e-1)
    labels = Symbol[]

    for Q in reps
        if lattice_isometry(Q, Q_hex; tol=tol)
            push!(labels, :HEX)
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
    hex_fraction = Float64[]
    # optional: store full info
    sizes_all = Vector{Vector{Int}}()
    labels_all = Vector{Vector{Symbol}}()
    min_eigs_all = Vector{Vector{Float64}}()

    for alpha in alphas_sorted
        data = results[alpha]
        total = sum(data.sizes)
        hex_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :HEX
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
        push!(hex_fraction, hex_size / total)
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
        hex_fraction = hex_fraction
    )
end

plot_data = prepare_plot_data(results_landscape_analysis)

@save joinpath(results_dir, "alpha_critical_analysis","temporary_results" ,"landscape_analysis.jld2") plot_data results_landscape_analysis

function compute_fractions(results)
    alphas_sorted = sort(collect(keys(results)))

    hex = Float64[]
    other = Float64[]

    for α in alphas_sorted
        data = results[α]

        total = sum(data.sizes) #total number of minimizers per alpha_i

        hex_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :HEX
        ; init=0)

        other_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :other
        ; init=0)

        push!(hex, total > 0 ? hex_size / total : 0.0)
        push!(other, total > 0 ? other_size / total : 0.0)
    end

    return alphas_sorted, hex, other
end

function compute_class_numbers(results)
    alphas_sorted = sort(collect(keys(results)))

    hex = Float64[]
    other = Float64[]

    for α in alphas_sorted
        data = results[α]

        total = sum(data.sizes)

        hex_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :HEX
        ; init=0)

        other_size = sum(
            size for (size, label) in zip(data.sizes, data.labels)
            if label == :other
        ; init=0)

        push!(hex, total > 0 ? hex_size  : 0.0)
        push!(other, total > 0 ? other_size  : 0.0)
    end

    return alphas_sorted, hex, other
end


_, hex_fraction, other_fraction = compute_fractions(results_landscape_analysis)
_, hex_numbers, other_numbers = compute_class_numbers(results_landscape_analysis)

n = 2
α = plot_data.alphas
fig_dir = joinpath(results_dir, "alpha_critical_analysis", "Figures")

# Plot 1
ymax = maximum(plot_data.num_clusters)
xmin = minimum(α)
xmax = maximum(α)
x_1 = xmax
x_6 = xmax
range = xmax - xmin
ticks = range / 20

p1 = plot(α, plot_data.num_clusters,
    xlabel="alpha",
    ylabel="# potential optimal lattices",
    marker=:o,
    color = :green,
    size = (800, 300),  # (width, height) → reduced height
    left_margin = 10mm,
    bottom_margin = 8mm,
    xlim=(xmin, x_1),
    ylim=(0, ymax + 1),
    yticks = 0:1:ymax+1,
    xticks = xmin:ticks:x_1,
    label = "Number of potential optimal lattices"
)
savefig(p1, joinpath(fig_dir, "num_minimizers.pdf"))



p3 = plot(α, plot_data.min_eig_global,
    xlabel="alpha",
    ylabel="Min EV of the \n Hessian matrix",
    marker=:o,
    color = :green,
    size = (800, 300),  # (width, height) → reduced height
    left_margin = 10mm,
    bottom_margin = 8mm,
    linewidth=1,
    xticks = xmin:0.5:xmax,
     xlim=(xmin, xmax),
    label="Stability of minimizers"
)

hline!(p3, [0], color=:red, linestyle=:dash, label="second-order criticality threshold")

savefig(p3, joinpath(fig_dir, "stability.pdf"))

# Plot 4
p4 = plot(α, hex_fraction,
    xlabel="alpha",
    ylabel="Fraction of runs",
    marker=:o,
    size = (800, 300),  # (width, height) → reduced height
    left_margin = 10mm,
    bottom_margin = 8mm,
    title="Fraction of runs converging to HEX-type minimizers",
    ylim=(0,1),
    label="HEX",
    xlim=(xmin, xmax),
    xticks = xmin:ticks:xmax)

savefig(p4, joinpath(fig_dir, "hex_fraction.pdf") )


# Plot 5
p = plot(α, hex_fraction,
    label="HEX",
    xlabel="alpha",
    ylabel="cluster hit probability",
    marker=:o,
    size = (800, 300),  # (width, height) → reduced height
    left_margin = 10mm,
    bottom_margin = 8mm,
    ylim=(0,1.1),
    xlim=(xmin, 2),
    xticks = xmin:(2/10):2,
    linewidth=1,
    legend = :bottomright
)


plot!(p, α, other_fraction,
    label="Other",
    marker=:o,
    linewidth=1
)

savefig(p, joinpath(fig_dir, "fraction_breakdown.pdf"))


# Plot 6: number of classes per type
p = plot(α, hex_numbers,
    label="HEX",
    xlabel="alpha",
    ylabel="Total cluster frequencies",
    size = (800, 300),
    left_margin = 10mm,
    bottom_margin = 8mm,
    marker=:o,
    ylim=(-0.1, maximum(hex_numbers) + 10),
    xlim=(xmin, 2),
    xticks = xmin:(2/10):2,
    linewidth=2,
    legend = :topleft 
)

plot!(p, α, other_numbers,
    label="Other",
    marker=:o,
    linewidth=2
)

savefig(p, joinpath(fig_dir, "number_breakdown.pdf"))   