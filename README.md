# Simpsons.jl
Julia module to check data for a Simpson's statistical paradox

## Usage

    using Simpsons
    
    has_simpsons_paradox(df, cause, effect, factor; continuous_threshold=5, verbose=true)
    
Returns true if the data aggregated by factor exhibits Simpson's paradox.
Note that the cause and effect columns must be numeric in type.
A continuous data factor (one with continuous_threshold or more discrete
levels) will be grouped into a binary factor so as to avoid too many clusters.
Prints the regression slope directions for overall data and groups if verbose is true.

    simpsons_analysis(df, cause_column, effect_column; verbose = true, show_plots = true)
    
Analyze the dataframe df assuming a cause is in cause_column and an effect in
effect_column of the dataframe. Output data including and Simpson's paradox type
reversals in subgroups found. Plots shown if show_plots is true (default).

    make_paradox(nsubgroups = 3 , N = 16000)
 
Return a dataframe containing random data in 3 columns :x (cause), :y (effect), and
:z (cofactor) which displays the Simpson's paradox.

    plot_clusters(df, cause, effect)
    
Plot, with subplots, clustering of the dataframe using cause and effect plotted and
color coded by clusterings. Use kmeans clustering analysis on all fields of
dataframe. Use 2 to 5 as cluster number. Ignores non-numeric columns.

    plot_kmeans_by_factor(df, cause_column, effect_column, factor_column)
    
Plot, clustering of the dataframe using cause as X, effect Y, with the factor_column
used for kmeans clustering into 2 clusters on the plot. The factor must be numeric.


### Examples

    using Simpsons
    
    # Create a dataframe with cause :x, effect :y, and cofactor :z columns
    dfp = make_paradox()
    
    # Test for a Simpson's paradox, where the regression direction :x with :y 
    #    reverses if the data is split by factor :z.
    has_simpsons_paradox(dfp, :x, :y, :z)  # true with this data

    # analyze with graphs of clustering. To see graphs, run in REPL to prevent premature display closure. 
    simpsons_analysis(dfp, :x, :y, verbose=false)
    

### Installation

Install the package using the package manager (] to enter pkg> mode):

    (v1) pkg> add Simpsons

