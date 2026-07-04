#!/usr/bin/env julia
"""Quick PNG export from Trixi VTU (ReadVTK + Plots)."""

using Pkg
Pkg.activate(@__DIR__)

for pkg in ("ReadVTK", "Plots")
    try
        eval(Meta.parse("using $pkg"))
    catch
        @info "Installing $pkg"
        Pkg.add(pkg)
    end
end
using ReadVTK, Plots

const OUT = normpath(joinpath(@__DIR__, "..", "..", "..", "outputs", "kelvin_helmholtz", "trixi_10steps"))
const VTU_DIR = joinpath(OUT, "vtu_converted")
const PNG_DIR = joinpath(OUT, "frames_jl")
mkpath(PNG_DIR)

vtu_files = sort(filter(f -> startswith(basename(f), "solution_") && endswith(f, ".vtu") && !contains(f, "celldata"),
                   readdir(VTU_DIR; join=true)))

function cell_centers(vtk)
    x = coordinates(vtk)[1]
    y = coordinates(vtk)[2]
    # approximate: use point coords averaged per cell via connectivity
    conn = connectivity(vtk)
    pts_x = vec(x)
    pts_y = vec(y)
    cx = Float64[]
    cy = Float64[]
    for c in eachindex(conn)
        ids = conn[c]
        push!(cx, mean(pts_x[ids]))
        push!(cy, mean(pts_y[ids]))
    end
    cx, cy
end

println(">>> Plotting ", length(vtu_files), " frames")
for (i, path) in enumerate(vtu_files)
    vtk = VTKFile(path)
    rho = vec(get_data(vtk, "rho"))
    cx, cy = cell_centers(vtk)
    step = split(basename(path), "_")[end][1:end-4]
    plt = scatter(cx, cy, zcolor=rho, markersize=1.5, markerstrokewidth=0,
                  xlims=(-1, 1), ylims=(-1, 1), aspect_ratio=:equal,
                  color=:balance, clims=(0.45, 1.95),
                  title="Trixi KH step=$step", xlabel="x", ylabel="y", legend=false)
    png = joinpath(PNG_DIR, @sprintf("frame_%04d.png", i - 1))
    savefig(plt, png)
    println("  ", basename(path), " -> ", basename(png))
end
println(">>> DONE ", PNG_DIR)
