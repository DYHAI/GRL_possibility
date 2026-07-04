#!/usr/bin/env julia
"""Export one member's cell-center fields (rho, v1, v2, p) to a per-member NPZ."""

using Pkg
Pkg.activate(@__DIR__)
using NPZ, Printf

include(joinpath(@__DIR__, "export_ensemble_4var.jl"))

out_dir = length(ARGS) >= 1 ? normpath(ARGS[1]) :
    normpath(joinpath(@__DIR__, "..", "..", "..", "outputs", "kelvin_helmholtz", "trixi_ensemble_200"))
member_id = length(ARGS) >= 2 ? parse(Int, ARGS[2]) :
    parse(Int, get(ENV, "KH_MEMBER_ID", "1"))

member_dir = joinpath(out_dir, "member_$(lpad(member_id, 3, '0'))")
isdir(member_dir) || error("Missing member dir: $member_dir")

println(">>> Export 4-var fields  member=$member_id  dir=$member_dir")
flush(stdout)

frames = export_member(member_dir, member_id)
t0_meta, t_end_meta, save_dt = read_ensemble_meta(out_dir)

dict = Dict{String,Any}()
dict["members"] = Int32[member_id]
dict["n_members"] = Int32(1)
dict["var_names"] = npz_encode_strings(["rho", "v1", "v2", "p"])
dict["align_frames"] = Int32(0)
dict["t0"] = Float32(t0_meta)
dict["t_end"] = Float32(t_end_meta)
if save_dt !== nothing
    dict["save_dt"] = Float32(save_dt)
    if !isempty(frames)
        dict["t0"] = Float32(frames[1]["time"])
    end
end
dict["n_frames_max"] = Int32(length(frames))
dict["n_frames"] = Int32(length(frames))
dict["member_$(member_id)_nframes"] = Int32(length(frames))
dict["member_$(member_id)_times"] = Float32[Float32(fr["time"]) for fr in frames]
for (fi, fr) in enumerate(frames)
    fi0 = fi - 1
    for var in ("rho", "v1", "v2", "p", "cx", "cy")
        dict["$(var)_$(member_id)_$(fi0)"] = fr[var]
    end
    dict["step_$(member_id)_$(fi0)"] = Int32(fr["step"])
    dict["time_$(member_id)_$(fi0)"] = Float32(fr["time"])
end

npz_out = joinpath(member_dir, "member_$(lpad(member_id, 3, '0'))_fields_4var.npz")
npzwrite(npz_out, dict)
println(">>> Wrote $npz_out  frames=$(length(frames))")
