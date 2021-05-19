using Simpsons, Distributions, DataFrames, CSV, Plots, Test

# see wikipedia entry at http://en.wikipedia.org/wiki/Simpsons_paradox
const data = vcat(
    repeat([["A", "small", 1]], 81), repeat([["A", "small", 0]], 87 - 81),
    repeat([["B", "small", 0]], 270 - 234), repeat([["B", "small", 1]], 234),
    repeat([["B", "large", 1]], 55), repeat([["B", "large", 0]], 80 - 55),
    repeat([["A", "large", 0]], 263 - 192), repeat([["A", "large", 1]], 192)
)
const df = DataFrame(treatment = [d[1] for d in data],
               recovery = [d[3] for d in data],
               kidney_stone_size = [d[2] for d in data])

@test has_simpsons_paradox(df, :treatment, :recovery, :kidney_stone_size) == true

const pathname = "cars.csv"
const dfc = DataFrame(CSV.File(pathname, datarow = 3))
@test has_simpsons_paradox(dfc, :Weight, :MPG, :Cylinders) == false
@test has_simpsons_paradox(dfc, :Weight, :MPG, :Acceleration) == false
@test has_simpsons_paradox(dfc, :Weight, :MPG, :Horsepower) == false

Plots.scalefontsizes(0.6)
simpsons_analysis(dfc, :Horsepower, :Acceleration)
simpsons_analysis(df, :treatment, :recovery)
simpsons_analysis(make_paradox(), :x, :y)
