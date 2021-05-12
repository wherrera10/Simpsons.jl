module Simpsons

export has_simpsons_paradox, plot_clusters, plot_by_factor, simpsons_analysis

using DataFrames, Polynomials, Clustering, Plots

"""
    tointvec(col::Vector)

Convert the elements of a vector such as a DataFrame column to integers, based on
their alphabetical order when converted to string. Returns a Vector{Int} if col
is not numeric in type. Returns a copy of col if col is numeric in type.
"""
function tointvec(col::Vector)
    eltype(col) <: Number && return copy(col)
    d = Dict(d[s] => i for (i, s) in sort(["$x" for x in col]))
    return map(s -> d[s], col)
end

"""
    has_simpsons_paradox(df, cause_column, effect_column, factor_column)
True if the data aggregated by factor exhibits Simpson's paradox.
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
        println("Overall linear trend from cause to effect is ",
            overallslope > 0 ? "positive." : "negative.")
        for (i, slp) in enumerate(subgroupslopes)
            println("Subgroup $i trend is ", slp > 0 ? "positive." : "negative.")
        end
    end
    return any(slp -> slp != overallslope, subgroupslopes)
end

"""
    plot_clusters(df, cause_column, effect_column, maxclusters=4)

Plot, with subplots, clustering of the dataframe using caue and effect plotted and
color coded by clusterings. Use kmeans cluster analysis on all fields of dataframe
for clusters of size 2 to maxclusters(default 4).
"""
function plot_clusters(df, cause_column, effect_column, maxclusters=4)
    # convert non-numeric columns to numeric ones
    df1 = deepcopy(df)
    for s in names(df1)
        df1[:, s] = tointvec(df1[!, s])
    end
    factors = collect(Matrix(df1)')
    zresults = [kmeans(factors, nclust).assignments for nclust in 2:maxclusters]
    plt = plot(df[!, cause_column], df[!, effect_column], marker_z = zresults,
        color=:lightrainbow, layout=(maxclusters-1, 1))
    display(plt)
end

"""
    plot_by_factor(df, cause_column, effect_column, factor_column)

Plot, clustering of the dataframe using cause as X, effect Y, with the factor_column
used for kmeans clustering into 2 clusters on the plot.
"""
function plot_by_factor(df, cause_column, effect_column, factor_column)
    df1 = df[:, [cause_column, effect_column, factor_column]]
    df1[:, factor_column] = tointvec(df1[!, factor_column])
    zresult = kmeans(factors, 2).assignments
    plt = plot(df[!, cause_column], df[!, effect_column], marker_z = zresult, color=:lightrainbow)
    display(plt)
end

"""
    simpsons_analysis(df, cause_column, effect_column, show_plots = true)

Analyze the dataframe df assuming a cause is in cause_column and an effect in
effect_column of the dataframe. Output data including and Simpson's paradox type
reversals in subgroups found. Plots shown if show_plots is true (default).
"""
function simpsons_analysis(df, cause_column, effect_column, show_plots = true)
    # get / plot cause effect / slope overall
    # plot clusterings
    # for each factor, show / plot clustering and print slope, whether has simpsons
end


end  # module Simpsons
