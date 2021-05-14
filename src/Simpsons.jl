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
function has_simpsons_paradox(df, cause_column, effect_column, factor_column, verbose=true)
    # check that the cause and effect column data types are numeric
    typeof(df[1, cause_column]) <: Number || error("Column $cause_column must be numeric")
    typeof(df[1, effect_column]) <: Number || error("Column $effect_column must be numeric")

    # Do linear regression on the cause versus effect columns.
    df1 = df[:, [cause_column, effect_column]]
    m = fit(df[!, effect_column], df[!, cause_column], 1)
    overallslope = m.coeffs[2]

    # Group by the factor_column and do a similar linear regression on each group
    grouped = groupby(df, factor_column)
    subgroupslopes = Float64[]
    for (i, gdf) in enumerate(grouped)
        gm = fit(gdf[!, effect_column], gdf[!, cause_column], 1)
        push!(subgroupslopes, gm.coeffs[2])
    end
    if verbose
        println("For cause $cause_column, effect $effect_column, and factor $factor_column:")
        println("Overall linear trend from cause to effect is ",
            overallslope > 0 ? "positive." : "negative.")
        for (i, slp) in enumerate(subgroupslopes)
            println("    Subgroup $i trend is ", slp > 0 ? "positive." : "negative.")
            if sign(slp) != sign(overallslope)
                println("        This shows a Simpson paradox type reversal.")
            end
        end
    end
    return any(slp -> slp != overallslope, subgroupslopes)
end

"""
    plot_clusters(df, cause_column, effect_column)

Plot, with subplots, clustering of the dataframe using cause and effect plotted and
color coded by clusterings. Use kmeans clustering analysis on all fields of
dataframe. Use 2 to 5 as cluster number. Ignores non-numeric columns.
"""
function plot_clusters(df, cause_column, effect_column)
    # convert non-numeric columns to numeric ones
    df1 = df[:, filter(s -> type(df[1, s]) <: Number, names(df))]
    factors = collect(Matrix(df1)')
    subplots = Plots.Plot[]
    for n in 2:5
        push!(subplots, scatter(df1[!, cause_column], df1[!, effect_column],
            marker_z = kmeans(factors, n).assignments, color = :lightrainbow))
    end
    plt = scatter(subplots..., layout = (2, 2))
    display(plt)
end

"""
    plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)

Plot, clustering of the dataframe using cause as X, effect Y, with the factor_column
used for kmeans clustering into 2 clusters on the plot.
"""
function plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)
    df1 = DataFrame(cause_column => df[!, cause_column], effect_column => df[!, effect_column], 
        factor_column => tointvec(df[!, factor_column]))
    zresult = kmeans(collect(Matrix(df1)'), 2).assignments
    plt = scatter(df[!, cause_column], df[!, effect_column], marker_z = zresult, color = :lightrainbow)
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
        show_plots && plot_kmeans_by_factor(df, cause_column, effect_column, factor)
        has_simpsons_paradox(df, cause_column, effect_column, factor, verbose)
    end
end

end  # module Simpsons
