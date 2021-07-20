## Simpsons.jl

[![CI](https://github.com/wherrera10/Simpsons.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/wherrera10/Simpsons.jl/actions/workflows/ci.yml)

Julia module to check data for a Simpson's statistical paradox
<br><br>

<img src="https://github.com/wherrera10/Simpsons.jl/blob/main/docs/src/simpsons_example_plot.svg">

### Usage

    using Simpsons
    
    has_simpsons_paradox(df, cause, effect, factor; continuous_threshold = 5, cmax = 5, verbose = true)

Returns true if the data in DataFrame `df` aggregated by `factor` exhibits
Simpson's paradox. Note that the `cause` and `effect` columns will be converted
to Int columns if they are not already numeric in type. A continuous data
`factor` column (one with `continuous_threshold` or more discrete levels) will
be grouped into at most `cmax` clusters so as to avoid too many clusters. Prints
the regression slope directions for overall data and groups if `verbose` is true.
<br><br><br>

    simpsons_analysis(df, cause_column, effect_column; verbose = true, show_plots = true)
    
Analyze the DataFrame `df` assuming a cause is in `cause_column` and an effect in
`effect_column` of the dataframe. Output data including any Simpson's paradox type
first degree slope reversals in subgroups found. Plots shown if `show_plots` is true (default).
<br><br><br>

    make_paradox(nsubgroups = 3 , N = 1024)

Return a dataframe containing `N` rows of random data in 3 columns `:x` (cause), 
`:y` (effect), and `:z` (cofactor) which displays the Simpson's paradox.
<br><br><br>

    plot_clusters(df, cause, effect)
    
Plot, with subplots, clustering of the dataframe `df` using `cause` and `effect` plotted and
color coded by clusterings. Use kmeans clustering analysis on all fields of dataframe.
Use 2 to 5 as cluster numbers.
<br><br><br>

    plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)

Plot clustering of the dataframe using cause plotted as X, effect as Y, with the `factor_column`
used for kmeans clustering into between 2 and 5 clusters on the plot.
<br><br><br>

    find_clustering_elbow(dataarray::AbstractMatrix{<:Real}, cmin = 1, cmax = 5; fclust = kmeans, kwargs...)

Find the "elbow" of the totalcost versus cluster number curve, where
cmin <= elbow <= cmax. Note that in pathological cases where the actual
minimum of the totalcosts occurs at a cluster count less than that of the
curve "elbow", the function will return either cmin or the actual cluster
count at which the totalcost is at minimum, whichever is larger.
<br>
Returns a tuple: the cluster count and the ClusteringResult at the "elbow" optimum.
<br><br><br>


### Examples

    using Simpsons
    
    # Create a dataframe with cause :x, effect :y, and cofactor :z columns
    dfp = make_paradox()
    
    # Test for a Simpson's paradox, where the regression direction of :x with :y 
    #    reverses if the data is split by factor :z.
    has_simpsons_paradox(dfp, :x, :y, :z)  # true with this data

    # Analyze with plots made of data clustering. 
    # To see the plots, run in REPL to prevent premature display closure. 
    simpsons_analysis(dfp, :x, :y)
<br><br>


### Installation

Install the package using the package manager (Press ] to enter pkg> mode):

    (v1) pkg> add Simpsons

