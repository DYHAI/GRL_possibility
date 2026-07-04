#!/usr/bin/env julia
"""Export all 4 primitive fields (rho, v1, v2, p) at cell centers for each member/frame."""

using Pkg
Pkg.activate(@__DIR__)
using ReadVTK, NPZ, Statistics, Trixi2Vtk, HDF5, Printf

out_dir = length(ARGS) >= 1 ? normpath(ARGS[1]) :
    normpath(joinpath(@__DIR__, "..", "..", "..", "outputs", "kelvin_helmholtz", "trixi_ensemble_branch"))
nvis = parse(Int, get(ENV, "NVIS", "8"))

function npz_encode_strings(names::AbstractVector{<:AbstractString})
    n = length(names)
    maxlen = maximum(length, names)
    mat = zeros(UInt8, n, maxlen)
    for (i, s) in enumerate(names)
        mat[i, 1:length(s)] .= codeunits(s)
    end
    mat
end

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

function export_member(member_dir::String, member_id::Int)
    vtu_dir = joinpath(member_dir, "vtu_converted")
    h5_dir = joinpath(member_dir, "vtu")
    rm(vtu_dir; recursive = true, force = true)
    mkpath(vtu_dir)

    h5_files = sort(filter(f -> startswith(f, "solution_") && endswith(f, ".h5"), readdir(h5_dir)))
    for f in h5_files
        trixi2vtk(joinpath(h5_dir, f); output_directory = vtu_dir, nvisnodes = nvis)
    end

    vtu_files = sort(filter(f -> startswith(basename(f), "solution_") && endswith(f, ".vtu") && !contains(f, "celldata"),
                           readdir(vtu_dir; join = true)))
    frames = Vector{Dict{String,Any}}()
    for (fi, path) in enumerate(vtu_files)
        vtk = VTKFile(path)
        pts = get_points(vtk)
        cells = get_cells(vtk)
        cd = get_cell_data(vtk)
        rho = Float32.(vec(get_data(cd["rho"])))
        v1 = Float32.(vec(get_data(cd["v1"])))
        v2 = Float32.(vec(get_data(cd["v2"])))
        p = Float32.(vec(get_data(cd["p"])))
        cx, cy = cell_centers(pts, cells)
        step = parse(Int, split(basename(path), "_")[end][1:end-4])
        t = HDF5.h5open(joinpath(h5_dir, @sprintf("solution_%09d.h5", step)), "r") do file
            Float32(read(attributes(file)["time"]))
        end
        push!(frames, Dict(
            "step" => step, "time" => Float64(t),
            "rho" => rho, "v1" => v1, "v2" => v2, "p" => p, "cx" => cx, "cy" => cy,
        ))
        println("  member=$member_id  step=$step  t=$(round(t, digits=4))  n=$(length(rho))")
    end
    frames
end

function read_ensemble_meta(out_dir::String)
    t0 = 0.0
    t_end = 5.0
    save_dt = nothing
    meta_path = joinpath(out_dir, "ensemble_meta.txt")
    isfile(meta_path) || return t0, t_end, save_dt
    for line in eachline(meta_path)
        if startswith(line, "t0=")
            t0 = parse(Float64, split(line, "=")[2])
        elseif startswith(line, "t_end=")
            t_end = parse(Float64, split(line, "=")[2])
        elseif startswith(line, "save_dt=")
            save_dt = parse(Float64, split(line, "=")[2])
        end
    end
    return t0, t_end, save_dt
end

time_index(t, t0, save_dt) = round(Int, round((t - t0) / save_dt))

const FIELD_VARS = ("rho", "v1", "v2", "p", "cx", "cy")

function export_aligned_4var!(dict, members, member_frames, save_dt)
    dict["align_frames"] = Int32(1)
    if save_dt !== nothing
        t0_align = dict["t0"]
        buckets = Dict{Int, Dict{Int, Dict{String,Any}}}()
        for mid in members
            for fr in member_frames[mid]
                ti = time_index(fr["time"], t0_align, save_dt)
                get!(buckets, ti, Dict{Int, Dict{String,Any}}())[mid] = fr
            end
        end
        common_ti = sort(collect(keys(buckets)))
        filter!(ti -> length(buckets[ti]) == length(members), common_ti)
        times = Float32[]
        for (fi, ti) in enumerate(common_ti)
            push!(times, Float32(t0_align + ti * save_dt))
            for mid in members
                fr = buckets[ti][mid]
                for var in FIELD_VARS
                    dict["$(var)_$(mid)_$(fi-1)"] = fr[var]
                end
            end
            ref = buckets[ti][members[1]]
            dict["step_$(fi-1)"] = Int32(ref["step"])
            dict["time_$(fi-1)"] = Float32(ref["time"])
        end
        dict["times"] = times
        dict["n_frames"] = Int32(length(common_ti))
    else
        all_n_frames = minimum(length(member_frames[mid]) for mid in members)
        dict["n_frames"] = Int32(all_n_frames)
        for mid in members
            for fi in 0:(all_n_frames-1)
                fr = member_frames[mid][fi+1]
                for var in FIELD_VARS
                    dict["$(var)_$(mid)_$(fi)"] = fr[var]
                end
                if mid == members[1]
                    dict["step_$(fi)"] = Int32(fr["step"])
                    dict["time_$(fi)"] = Float32(fr["time"])
                end
            end
        end
        dict["times"] = Float32[dict["time_$(fi)"] for fi in 0:(all_n_frames-1)]
    end
    for mid in members
        dict["member_$(mid)_nframes"] = Int32(length(member_frames[mid]))
    end
end

function export_per_member_4var!(dict, members, member_frames)
    dict["align_frames"] = Int32(0)
    n_max = maximum(length(member_frames[mid]) for mid in members)
    dict["n_frames_max"] = Int32(n_max)
    dict["n_frames"] = Int32(n_max)
    for mid in members
        frames = member_frames[mid]
        nf = length(frames)
        dict["member_$(mid)_nframes"] = Int32(nf)
        dict["member_$(mid)_times"] = Float32[Float32(fr["time"]) for fr in frames]
        for (fi, fr) in enumerate(frames)
            fi0 = fi - 1
            for var in FIELD_VARS
                dict["$(var)_$(mid)_$(fi0)"] = fr[var]
            end
            dict["step_$(mid)_$(fi0)"] = Int32(fr["step"])
            dict["time_$(mid)_$(fi0)"] = Float32(fr["time"])
        end
    end
end

function main_export_ensemble(out_dir::String)
    member_dirs = sort(filter(d -> startswith(d, "member_"), readdir(out_dir)))
    isempty(member_dirs) && error("No member_* dirs in $out_dir")

    align_frames = get(ENV, "KH_EXPORT_ALIGN", "0") != "0"

    members = Int32[]
    member_frames = Dict{Int, Vector{Dict{String,Any}}}()
    for d in member_dirs
        mid = parse(Int, d[8:end])
        push!(members, Int32(mid))
        member_frames[mid] = export_member(joinpath(out_dir, d), mid)
    end

    t0_meta, t_end_meta, save_dt = read_ensemble_meta(out_dir)
    dict = Dict{String,Any}()
    dict["members"] = members
    dict["n_members"] = Int32(length(members))
    dict["var_names"] = npz_encode_strings(["rho", "v1", "v2", "p"])
    dict["t0"] = Float32(t0_meta)
    dict["t_end"] = Float32(t_end_meta)
    if save_dt !== nothing
        dict["save_dt"] = Float32(save_dt)
        if !isempty(member_frames[members[1]])
            dict["t0"] = Float32(member_frames[members[1]][1]["time"])
        end
    end

    if align_frames
        export_aligned_4var!(dict, members, member_frames, save_dt)
    else
        export_per_member_4var!(dict, members, member_frames)
    end

    npz_out = joinpath(out_dir, "ensemble_fields_4var.npz")
    npzwrite(npz_out, dict)
    mode = align_frames ? "aligned" : "per_member"
    println(">>> Wrote $npz_out  mode=$mode  members=$(members)  frames=$(dict["n_frames"])" *
            (save_dt === nothing ? "" : "  save_dt=$save_dt"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_export_ensemble(out_dir)
end
