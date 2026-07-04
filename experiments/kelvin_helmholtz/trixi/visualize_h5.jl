#!/usr/bin/env julia
"""Convert Trixi HDF5 snapshots to VTU for visualization."""

using Pkg
Pkg.activate(@__DIR__)

try
    using Trixi2Vtk
catch
    @info "Installing Trixi2Vtk (one-time)..."
    Pkg.add("Trixi2Vtk")
    using Trixi2Vtk
end

const OUT = normpath(joinpath(@__DIR__, "..", "..", "..", "outputs", "kelvin_helmholtz", "trixi_10steps"))
const H5_DIR = joinpath(OUT, "vtu")
const VTU_DIR = joinpath(OUT, "vtu_converted")

mkpath(VTU_DIR)

h5_files = sort(filter(f -> startswith(basename(f), "solution_") && endswith(f, ".h5"),
                   readdir(H5_DIR; join=true)))
isempty(h5_files) && error("No solution_*.h5 in $H5_DIR")

println(">>> Converting ", length(h5_files), " HDF5 → VTU (nvisnodes=8)")
for f in h5_files
    trixi2vtk(f; output_directory=VTU_DIR, nvisnodes=8)
    println("  ", basename(f))
end
println(">>> DONE VTU_DIR=", VTU_DIR)
