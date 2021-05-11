module Simpsons

export has_simpsons_paradox

using DataFrames, Polynomials

"""
    has_simpsons_paradox(df, cause_column, effect_column, factor_column)

True if the data ggregated by factor exhibits Simpson's paradox.
example:
    df = DataFrame(
        treatment = ["A", "B", "A", "A", "B", "B"],
        recovery = [1, 0, 1, 1, 0, 0],
        kidney_stone_size = ["small", "small", "large", "small", "large", "large"])

   simpsons_paradox(df, "treatment", "recovery", "kidney_stone_size")
"""
function has_simpsons_paradox(df, cause_column, effect_column, factor_column, verbose=true)
    # first do a regression on the cause versus effect columns.
    df1 = df[:, [cause_column, effect_column]]
    m = fit(df[!, effect_column], df[!, cause_column], 1)
    overallslope = sign(m.coeffs[2])
    # next group by the factor column and do a regression on each group
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

end  # module Simpsons
