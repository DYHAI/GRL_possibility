#!/usr/bin/env julia
"""Export VTU cell data to NPZ for fast Python plotting."""

using Pkg
Pkg.activate(@__DIR__)
using ReadVTK, NPZ, Statistics

out_dir = length(ARGS) >= 1 ? normpath(ARGS[1]) :
    normpath(joinpath(@__DIR__, "..", "..", "..", "outputs", "kelvin_helmholtz", "trixi_10steps"))
nvis = parse(Int, get(ENV, "NVIS", "1"))  # 1 = native cell resolution (1024² sim needs no upsample)
vtu_dir = joinpath(out_dir, "vtu_converted")
if !isdir(vtu_dir) || isempty(filter(f -> endswith(f, ".vtu"), readdir(vtu_dir)))
    h5_dir = joinpath(out_dir, "vtu")
    vtu_dir = joinpath(out_dir, "vtu_converted")
    mkpath(vtu_dir)
    using Trixi2Vtk
    h5_files = sort(filter(f -> startswith(f, "solution_") && endswith(f, ".h5"), readdir(h5_dir)))
    if !isempty(h5_files)
        println(">>> Converting ", length(h5_files), " H5 -> VTU (nvisnodes=", nvis, ")")
        for f in h5_files
            trixi2vtk(joinpath(h5_dir, f); output_directory=vtu_dir, nvisnodes=nvis)
        end
    end
end
npz_out = joinpath(out_dir, "snapshots.npz")

vtu_files = sort(filter(f -> startswith(basename(f), "solution_") && endswith(f, ".vtu") && !contains(f, "celldata"),
                   readdir(vtu_dir; join=true)))
isempty(vtu_files) && error("No VTU files in $vtu_dir")

function cell_centers(pts, cells)
    n = length(cells)
    cx = Vector{Float32}(undef, n)
    cy = Vector{Float32}(undef, n)
    off0 = 0
    for i in 1:n
        off1 = cells.offsets[i]
        ids = cells.connectivity[off0+1:off1]
        cx[i] = mean(@view pts[2, ids])
        cy[i] = mean(@view pts[1, ids])
        off0 = off1
    end
    cx, cy
end

dict = Dict{String,Any}()
steps = Int32[]

for path in vtu_files
    vtk = VTKFile(path)
    pts = get_points(vtk)
    cells = get_cells(vtk)
    cd = get_cell_data(vtk)
    rho = Float32.(vec(get_data(cd["rho"])))
    cx, cy = cell_centers(pts, cells)
    step = parse(Int32, split(basename(path), "_")[end][1:end-4])
    push!(steps, step)
    dict["rho_$step"] = rho
    dict["cx_$step"] = cx
    dict["cy_$step"] = cy
    println("  step=$step  n=$(length(rho))")
end
dict["steps"] = steps
npzwrite(npz_out, dict)
println(">>> Wrote ", npz_out)
