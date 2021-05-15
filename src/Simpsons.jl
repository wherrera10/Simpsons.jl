module Simpsons

export has_simpsons_paradox, plot_clusters, plot_kmeans_by_factor, simpsons_analysis

using DataFrames, Polynomials, Clustering, Plots

"""
    has_simpsons_paradox(df, cause_column, effect_column, factor_column, verbose=true)
Returns true if the data aggregated by factor exhibits Simpson's paradox.
Note that the cause_column and effect_column must be numeric in type.
example:
    df = DataFrame(
        treatment = [1, 2, 1, 1, 2, 2],
        recovery = [1, 0, 1, 1, 0, 0],
        kidney_stone_size = ["small", "small", "large", "small", "large", "large"])
   has_simpsons_paradox(df, :treatment, :recovery, :kidney_stone_size)
"""
function has_simpsons_paradox(df, cause, effect, factor, continuous_threshold=5, verbose=true)
    # check that the cause and effect column data types are numeric
    df[1, cause] isa Number || error("Column $cause must be numeric : $(df[1, cause_column])")
    df[1, effect] isa Number || error("Column $effect must be numeric")

    # Do linear regression on the cause versus effect columns.
    df1 = df[:, [cause, effect]]
    m = fit(df[!, effect], df[!, cause], 1)
    overallslope = m.coeffs[2]

    # Group by the factor_column and do a similar linear regression on each group when possible
    # first check for continous effect type, if number of unique values > continuous_threshold
    df1 = df[:, [cause, effect, factor]]
    eff = df1[!, effect]
    uni = unique(eff)
    if length(uni) >= continuous_threshold
        groupmat = zeros(eltype(uni), (2, length(eff)))
        groupmat[1, :] .= eff
        kr = kmeans(groupmat, 2)
        grou = Symbol("grouped" * string(effect))
        df1[:, grou] = kr.assignments
        grouped = groupby(df1, grou)
    else
        grouped = groupby(df1, factor)
    end
    subgroupslopes = Float64[]
    for (i, gdf) in enumerate(grouped)
        length(gdf[!, effect]) < 2 && continue
        gm = fit(gdf[!, effect], gdf[!, cause], 1)
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
    return differentslopes
end

"""
    plot_clusters(df, cause, effect)

Plot, with subplots, clustering of the dataframe using cause and effect plotted and
color coded by clusterings. Use kmeans clustering analysis on all fields of
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

Plot, clustering of the dataframe using cause as X, effect Y, with the factor_column
used for kmeans clustering into 2 clusters on the plot. The factor must be numeric.
"""
function plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)
    df[1, factor_column] isa Number || error("error_column must be numeric")
    df1 = DataFrame(cause_column => df[!, cause_column], effect_column => df[!, effect_column],
        factor_column => df[!, factor_column])
    zresult = kmeans(collect(Matrix(df1)'), 2).assignments
    plt = scatter(df[!, cause_column], df[!, effect_column], marker_z = zresult, color = :lightrainbow,
        title = "$cause_column -> $effect_column with cofactor $factor_column",
        xlabel = cause_column, ylabel=effect_column)
    display(plt)
end

"""
    simpsons_analysis(df, cause_column, effect_column, verbose = true, show_plots = true)

Analyze the dataframe df assuming a cause is in cause_column and an effect in
effect_column of the dataframe. Output data including and Simpson's paradox type
reversals in subgroups found. Plots shown if show_plots is true (default).
"""
function simpsons_analysis(df, cause_column, effect_column, verbose=true, show_plots = true)
    # Plot cluster analysis for clustering numbers 2 through 5
    show_plots && plot_clusters(df, cause_column, effect_column)
    # plot clusterings by factor
    for factor in filter(f -> !(f in [cause_column, effect_column]), names(df))
        if show_plots && df[1, factor] isa Number
            plot_kmeans_by_factor(df, cause_column, effect_column, factor)
        end
        has_simpsons_paradox(df, cause_column, effect_column, factor, verbose)
    end
end

end  # module Simpsons
