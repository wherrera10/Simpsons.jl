module Simpsons

export has_simpsons_paradox, make_paradox, plot_clusters, plot_kmeans_by_factor, simpsons_analysis

using DataFrames, Distributions, Polynomials, Clustering, Plots

"""
    has_simpsons_paradox(df, cause, effect, factor;
        continuous_threshold = 5, cmax = 5, verbose = true)
Returns true if the data aggregated by `factor` exhibits Simpson's paradox.
Note that the `cause` and `effect` columns will be converted to Int columns if
they are not already numeric in type. A continuous data `factor` column (one
with `continuous_threshold` or more discrete levels) will be grouped into a
at most cmax clusters so as to avoid too many clusters. Prints the regression
slope directions for overall data and groups if verbose is true.
Example:
    df = DataFrame(
        treatment = [1, 2, 1, 1, 2, 2],
        recovery = [1, 0, 1, 1, 0, 0],
        kidney_stone_size = ["small", "small", "large", "small", "large", "large"])
    has_simpsons_paradox(df, :treatment, :recovery, :kidney_stone_size)
"""
function has_simpsons_paradox(df, cause, effect, factor; continuous_threshold = 5, cmax = 5, verbose = true)
    df1 = DataFrame()

    # Convert cause and effect column data types to numeric if needed
    df1[:, cause] = (df[1, cause] isa Number ? df[!, cause] : _makenumeric(df[!, cause]))
    df1[:, effect] = (df[1, effect] isa Number ? df[!, effect] : _makenumeric(df[!, effect]))
    if df[1, factor] isa Number || length(unique(df[!, factor])) < continuous_threshold
        df1[:, factor] = df[!, factor]
    else
        df1[:, factor] = _makenumeric(df[!, factor])
    end

    # Do linear regression on the cause versus effect columns.
    m = Polynomials.fit(df1[!, effect], df1[!, cause], 1)
    overallslope = m.coeffs[2]

    # Group by the factor and do a similar linear regression on each group when possible
    # first check for continous factor type, if number of unique values > continuous_threshold
    fac = df1[!, factor]
    uni = unique(fac)
    if length(uni) >= continuous_threshold && uni[1] isa Number
        # continuous factor, so find best cluster number up to cmax
        groupmat = zeros(eltype(uni), (2, length(fac)))
        groupmat[1, :] .= fac
        _, kmclust = find_clustering_elbow(groupmat, 2, cmax)
        grou = Symbol("grouped" * string(factor))
        insertcols!(df1, grou => kmclust.assignments)
        grouped = groupby(df1, grou)
    else
        grouped = groupby(df1, factor)
    end
    subgroupslopes = Float64[]
    for gdf in grouped
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
    differentslopesigns = false
    for (i, slp) in enumerate(subgroupslopes)
        verbose && println("    Subgroup $i trend is ", slp > 0 ? "positive." : "negative.")
        if sign(slp) != sign(overallslope)
            verbose && println("        This shows a Simpson paradox type reversal.")
            differentslopesigns = true
        end
    end
    verbose && println()
    return differentslopesigns
end

"""
    make_paradox(nsubgroups = 3 , N = 1024)
Return a dataframe containing `N` rows of random data in 3 columns `:x` (cause), 
`:y` (effect), and `:z` (cofactor) which displays the Simpson's paradox.
"""
function make_paradox(nsubgroups = 3 , N = 1024)
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
dataframe. Use 2 to 5 as cluster number. Converts non-numeric columns to numeric for processing.
"""
function plot_clusters(df, cause, effect)
    # convert non-numeric columns to numeric ones
    df1 = DataFrame()
    for nam in names(df)
        df1[:, nam] = df[1, nam] isa Number ? df[!, nam] : _makenumeric(df[!, nam])
    end
    factors = collect(Matrix(df1)')
    subplots = Plots.Plot[]
    for n in 2:5
        push!(subplots, scatter(df1[!, cause], df1[!, effect],
            marker_z = kmeans(factors, n).assignments, color = :lightrainbow,
            title = "$cause -> $effect with $n clusters", legend = false,
            xlabel = cause, ylabel = effect, smooths = true))
    end
    plt = scatter(subplots..., layout = (2, 2), smooths=true)
    display(plt)
end

"""
    plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)
Plot clustering of the dataframe using cause plotted as X, effect as Y, with the `factor_column`
used for kmeans clustering into between 2 and 5 clusters on the plot.
"""
function plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)
    df1 = DataFrame(cause_column => df[!, cause_column], effect_column => df[!, effect_column],
        factor_column => df[1, factor_column] isa Number ? df[!, factor_column] : _makenumeric(df[!, factor_column]))
    fac = df1[!, factor_column]
    uni = unique(fac)
    groupmat = zeros(eltype(uni), (2, length(fac)))
    groupmat[1, :] .= fac
    karray = [kmeans(groupmat, i) for i in 1:6]
    x1, y1 = 1, karray[1].totalcost
    x2, y2 = 6, karray[6].totalcost
    (_, idx) = findmax(map(i -> distance(x1, y1, x2, y2, i, karray[i].totalcost), 2:5))
    grou = Symbol("grouped" * string(factor_column))
    df1[:, grou] = karray[idx + 1].assignments
    groups = groupby(df1, grou)
    plt = scatter(df1[!, cause_column], df[!, effect_column], color = :black, smooth = true,
        title = "$cause_column -> $effect_column with cofactor $factor_column",
        xlabel = cause_column, ylabel = effect_column, legend = false)
    for (i, gf) in enumerate(groups)
        scatter!(plt, gf[!, cause_column], gf[!, effect_column], color = _pcolor(i), smooth = true, legend = false)
    end
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

"""
    find_clustering_elbow(dataarray, cmin = 1, cmax = 5)

Find the "elbow" of the totalcost versus cluster number curve, where
cmin <= elbow <= cmax. Note that in pathological cases where the actual
minimum of the totalcosts occurs at a cluster count less than that of the
curve "elbow", the function will return either cmin or the actual cluster
count at which the totalcost is at minimum, whichever is larger.
<br>
Returns a tuple: the cluster count and the KmeansResult at the "elbow" optimum.
"""
function find_clustering_elbow(dataarray::AbstractMatrix{<:Real}, cmin = 1, cmax = 5)
    allkmeans = [kmeans(dataarray, i) for i in 1:cmax+1]
    alltotals = map(x -> x.totalcost, allkmeans)
    totalsmin, cidx = findmin(alltotals)
    x1, y1 = 1, alltotals[1]
    x2, y2 = cmax + 1, alltotals[cmax + 1]
    _, idx = findmax(map(i -> distance(x1, y1, x2, y2, i, alltotals[i]), 2:cmax))
    nclust = cidx < idx + 1 ? max(cmin, cidx) : idx + 1
    return nclust, allkmeans[nclust]
end


# internal helper functions

""" Make a column numeric """
function _makenumeric(a)
    d = Dict{eltype(a), Int}()
    for (i, s) in enumerate(sort(unique(a)))
        d[s] = i
    end
    return map(x -> d[x], a)
end

""" distance point p = [x0, y0] to line (L1 = [x1, y1], L2 = [x2, y2]), 2D """
function distance(x1, y1, x2, y2, x0, y0)
    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / sqrt((x2 - x1)^2 + (y2 - y1)^2)
end

_pcolor(i) = collect(palette(:lightrainbow))[mod1(i, 6)]

end  # module Simpsons
