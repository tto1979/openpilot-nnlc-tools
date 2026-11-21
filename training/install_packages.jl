#!/usr/bin/env julia
# Install and precompile all Julia packages needed for NNLC training.
# Used during Docker build to avoid a 20-minute wait on first run.

import Pkg

packages = [
    "CSV",
    "DataFrames",
    "StatsBase",
    "Flux",
    "MLDataUtils",
    "MLUtils",
    "Statistics",
    "LinearAlgebra",
    "PyFormattedStrings",
    "Random",
    "ProgressMeter",
    "Zygote",
    "Plots",
    "BSON",
    "CategoricalArrays",
    "SharedArrays",
    "SplitApplyCombine",
    "InvertedIndices",
    "JSON",
    "Dates",
    "CUDA",
    "ArgParse",
    "TeeStreams",
]

Pkg.add(packages)

# Current NVIDIA drivers (CUDA 13, also what RTX 50 cards come with) are refused by CUDA.jl 5.x
# ("only supports NVIDIA drivers for CUDA 11.x or 12.x"). Insist on 6.x; if something holds it back, fail here.
Pkg.add(Pkg.PackageSpec(name="CUDA", version="6"))
Pkg.status(["CUDA", "Flux"])

# Trigger precompilation
println("Precompiling all packages...")
for pkg in packages
    try
        Core.eval(Main, Meta.parse("using $pkg"))
        println("  ✓ $pkg")
    catch e
        println("  ⚠ $pkg ($(typeof(e)) — may work at runtime)")
    end
end

println("Package installation complete.")
