using Simpsons, DataFrames, CSV
using Test

# see wikipedia entry at http://en.wikipedia.org/wiki/Simpsons_paradox
const data = vcat(
    repeat([["A", "small", 1]], 81), repeat([["A", "small", 0]], 87 - 81),
    repeat([["B", "small", 0]], 270 - 234), repeat([["B", "small", 1]], 234),
    repeat([["B", "large", 1]], 55), repeat([["B", "large", 0]], 80 - 55),
    repeat([["A", "large", 0]], 263 - 192), repeat([["A", "large", 1]], 192)
)
const df = DataFrame(Atreatment = [d[1] == "A" ? 1 : 0 for d in data],
               recovery = [d[3] for d in data],
               kidney_stone_size = [d[2] for d in data])

const result = has_simpsons_paradox(df, :Atreatment, :recovery, :kidney_stone_size, true)

@test result == true

const pathname = download("https://perso.telecom-paristech.fr/eagan/class/igr204/data/cars.csv")
const dfc = DataFrame(CSV.File(pathname, datarow=3))
simpsons_analysis(dfc, :MPG, :Horsepower)
