#!/usr/bin/env julia
"""Smoke test: single KH AMR run (traj 1)."""

include(joinpath(@__DIR__, "kh_common.jl"))

out_dir = joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi")
@info "Output directory" out_dir
sol, meta = run_kh_trajectory(1, 1, out_dir; save_interval=100)
@info "Done." meta.retcode out_dir
