#!/usr/bin/env julia
"""Run one full KH rollout: t=0..3, seed selects IC perturbation."""

include(joinpath(@__DIR__, "kh_common.jl"))

traj_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1
seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : traj_id
save_interval = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50

out_root = joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "ensemble")
out_dir = joinpath(out_root, "traj_" * lpad(string(traj_id), 3, '0'))

@info "Starting trajectory" traj_id seed out_dir
sol, meta = run_kh_trajectory(traj_id, seed, out_dir; save_interval=save_interval)
@info "Finished" meta.retcode meta.elapsed_sec out_dir
