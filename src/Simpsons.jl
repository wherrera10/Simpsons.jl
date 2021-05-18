module Simpsons

export has_simpsons_paradox, make_paradox, plot_clusters, plot_kmeans_by_factor, simpsons_analysis

using DataFrames, Distributions, Polynomials, Clustering, Plots

"""
    has_simpsons_paradox(df, cause, effect, factor; continuous_threshold=5, verbose=true)

Returns true if the data aggregated by `factor` exhibits Simpson's paradox.
Note that the `cause` and `effect` columns must be numeric in type.
A continuous data `factor` column (one with `continuous_threshold` or more discrete
levels) will be grouped into a binary factor so as to avoid too many clusters.
Prints the regression slope directions for overall data and groups if verbose is true.

Example:
    df = DataFrame(
        treatment = [1, 2, 1, 1, 2, 2],
        recovery = [1, 0, 1, 1, 0, 0],
        kidney_stone_size = ["small", "small", "large", "small", "large", "large"])
    has_simpsons_paradox(df, :treatment, :recovery, :kidney_stone_size)
"""
function has_simpsons_paradox(df, cause, effect, factor; continuous_threshold=5, verbose=true)
    # check that the cause and effect column data types are numeric
    df[1, cause] isa Number || error("Column $cause must be numeric")
    df[1, effect] isa Number || error("Column $effect must be numeric")

    # Do linear regression on the cause versus effect columns.
    df1 = df[:, [cause, effect]]
    m = Polynomials.fit(df[!, effect], df[!, cause], 1)
    overallslope = m.coeffs[2]

    # Group by the factor and do a similar linear regression on each group when possible
    # first check for continous factr type, if number of unique values > continuous_threshold
    df1 = df[:, [cause, effect, factor]]
    fac = df1[!, factor]
    uni = unique(fac)
    if length(uni) >= continuous_threshold && uni[1] isa Number
        groupmat = zeros(eltype(uni), (2, length(fac)))
        groupmat[1, :] .= fac
        kr = kmeans(groupmat, 2)
        grou = Symbol("grouped" * string(factor))
        df1[:, grou] = kr.assignments
        grouped = groupby(df1, grou)
    else
        grouped = groupby(df1, factor)
    end
    subgroupslopes = Float64[]
    for (i, gdf) in enumerate(grouped)
        length(gdf[!, effect]) < 2 && continue
        gm = Polynomials.fit(gdf[!, effect], gdf[!, cause], 1)
        length(gm.coeffs) < 2 && continue
        push!(subgroupslopes, gm.coeffs[2])
    end
    if verbose
        println("For cause $cause, effect $effect, and factor $factor:")
        println("Overall linear trend from cause to effect is ",
            overallslope > 0 ? "positive." : "negative.")
    end
    differentslopes = false
    for (i, slp) in enumerate(subgroupslopes)
        verbose && println("    Subgroup $i trend is ", slp > 0 ? "positive." : "negative.")
        if sign(slp) != sign(overallslope)
            verbose && println("        This shows a Simpson paradox type reversal.")
            differentslopes = true
        end
    end
    verbose && println()
    return differentslopes
end

"""
    make_paradox(nsubgroups = 3 , N = 16000)

Return a dataframe containing random data in 3 columns `:x` (cause), `:y` (effect), and
`:z` (cofactor) which displays the Simpson's paradox.
"""
function make_paradox(nsubgroups = 3 , N = 16000)
    rw = rand(nsubgroups)
    w = rw ./ sum(rw)
    m = rand(MvNormal([0, 0], 3 .* [1 0.7; 0.7 1]), nsubgroups)
    cv = [[1 -c; -c 1] for c in rand(Uniform(0.1, 0.9), nsubgroups)]

    dfs = DataFrame(:x => Float64[], :y => Float64[], :z => Int[])
    for subgroup in 1:nsubgroups
        subN = Int(round(N .* w[subgroup]))
        xarr, yarr = Float64[], Float64[]
        for _ in 1:subN
            x, y = rand(MvNormal(m[:, subgroup], cv[subgroup]), 2)
            push!(xarr, x)
            push!(yarr, y)
        end
        samp = DataFrame(:x => xarr, :y => yarr, :z => fill(subgroup, subN))
        append!(dfs, samp)
    end
    return has_simpsons_paradox(dfs, :x, :y, :z, verbose=false) ? dfs : make_paradox(nsubgroups, N)
end

"""
    plot_clusters(df, cause, effect)

Plot, with subplots, clustering of the dataframe using `cause` (X axis) and `effect` (Y axis) 
plotted and color coded by clusterings. Use kmeans clustering analysis on all fields of
dataframe. Use 2 to 5 as cluster number. Ignores non-numeric columns.
"""
function plot_clusters(df, cause, effect)
    # convert non-numeric columns to numeric ones
    df1 = df[:, filter(s -> df[1, s] isa Number, names(df))]

    factors = collect(Matrix(df1)')
    subplots = Plots.Plot[]
    for n in 2:5
        push!(subplots, scatter(df1[!, cause], df1[!, effect],
            marker_z = kmeans(factors, n).assignments, color = :lightrainbow,
            title = "$cause -> $effect with $n clusters",
            xlabel = cause, ylabel=effect))
    end
    plt = scatter(subplots..., layout = (2, 2))
    display(plt)
end

"""
    plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)

Plot clustering of the dataframe using cause plotted as X, effect as Y, with the `factor_column`
used for kmeans clustering into 2 clusters on the plot. The factor must be numeric.
"""
function plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)
    df[1, factor_column] isa Number || error("Error: column $factor_column must be numeric")
    df1 = DataFrame(cause_column => df[!, cause_column], effect_column => df[!, effect_column],
        factor_column => df[!, factor_column])
    zresult = kmeans(collect(Matrix(df1)'), 2).assignments
    plt = scatter(df[!, cause_column], df[!, effect_column], marker_z = zresult, color = :lightrainbow,
        title = "$cause_column -> $effect_column with cofactor $factor_column",
        xlabel = cause_column, ylabel=effect_column)
    display(plt)
end

"""
    simpsons_analysis(df, cause_column, effect_column; verbose = true, show_plots = true)

Analyze the dataframe `df` assuming a cause is in `cause_column` and an effect in
`effect_column` of the dataframe. Output data including any Simpson's paradox type
reversals in subgroups found. Plots shown if show_plots is true (default).
"""
function simpsons_analysis(df, cause_column, effect_column; verbose=true, show_plots = true)
    # Plot cluster analysis for clustering numbers 2 through 5
    show_plots && plot_clusters(df, cause_column, effect_column)
    # plot clusterings by factor
    for factor in Symbol.(names(df))
        factor in [cause_column, effect_column] && continue
        if show_plots && df[1, factor] isa Number
            plot_kmeans_by_factor(df, cause_column, effect_column, factor)
        end
        has_simpsons_paradox(df, cause_column, effect_column, factor, verbose=verbose)
    end
end

end  # module Simpsons
