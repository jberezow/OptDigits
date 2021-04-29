using Pkg

metadata_packages = [
    "LinearAlgebra",
    "Random",
    "Distributions",
    "Distances",
    "Flux",
    "JLD",
    "MultivariateStats",
    "Serialization",
    "StatsBase"]

Pkg.add(Pkg.PackageSpec(;name="Gen",version="0.4.1"))
Pkg.add(Pkg.PackageSpec(;name="ReverseDiff",version="1.4.3"))

for package=metadata_packages
    Pkg.add(package)
end

app_dir = "/app"
push!(LOAD_PATH, app_dir)

using Gen
using LinearAlgebra
using Random
using Distributions
using Distances
using Flux
using JLD
using MultivariateStats
using Serialization
using StatsBase

