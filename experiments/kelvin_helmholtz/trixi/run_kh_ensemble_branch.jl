#!/usr/bin/env julia
"""
Stochastic ensemble from t=0:
  N members, same IC (seed=1), per-step random perturbation, integrate to T_END (default 5 s).
"""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const REF = parse(Int, get(ENV, "KH_REF", "5"))
const USE_AMR = get(ENV, "KH_AMR", "1") != "0"
const T_END = parse(Float64, get(ENV, "KH_T_END", "5.0"))
const IC_SEED = parse(Int, get(ENV, "KH_IC_SEED", "1"))
const N_MEMBERS = parse(Int, get(ENV, "KH_N_MEMBERS", "5"))
const SAVE_INTERVAL = parse(Int, get(ENV, "KH_SAVE_INTERVAL", "40"))
const SAVE_DT = let s = get(ENV, "KH_SAVE_DT", "0.1"); isempty(strip(s)) ? nothing : parse(Float64, s) end
const STEP_REL_EPS = parse(Float64, get(ENV, "KH_STEP_REL_EPS", "3e-5"))
const OUT = get(ENV, "KH_OUT", joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_ensemble_branch"))

n_cells_max = USE_AMR ? 100_000 : (2^(2 * REF) + 50_000)

mkpath(OUT)
if get(ENV, "KH_CLEAN", "0") != "0"
    for sub in readdir(OUT)
        p = joinpath(OUT, sub)
        if isdir(p) && (startswith(sub, "member_") || sub in ("baseline", "frames"))
            rm(p; recursive = true, force = true)
        end
    end
    for f in ("ensemble_snapshots.npz", "ensemble_fields_4var.npz", "ensemble_meta.txt",
              "gif_summary.json", "kh_ensemble_branch.gif", "unet_grids_512x4_summary.json")
        rm(joinpath(OUT, f); force = true)
    end
end

println(">>> Stochastic ensemble  t=0..$(T_END)  members=$N_MEMBERS")
println(">>>   IC seed=$IC_SEED (same for all)  step_rel_eps=$STEP_REL_EPS")
if SAVE_DT !== nothing
    println(">>>   save_dt=$(SAVE_DT) s (fixed physical time output)")
else
    println(">>>   save_interval=$(SAVE_INTERVAL) steps")
end
println(">>>   OUT=$OUT")
flush(stdout)

member_meta = []
for m in 1:N_MEMBERS
    perturb_seed = 1000 + m
    member_dir = joinpath(OUT, "member_$(lpad(m, 3, '0'))")
    println("\n>>> Member $m  perturb_seed=$perturb_seed")
    flush(stdout)
    _, meta = run_kh_member_stochastic(
        m, perturb_seed, T_END, member_dir;
        ic_seed = IC_SEED,
        save_interval = SAVE_INTERVAL,
        save_dt = SAVE_DT,
        step_rel_eps = STEP_REL_EPS,
        refinement_level = REF,
        n_cells_max = n_cells_max,
        enable_amr = USE_AMR,
    )
    push!(member_meta, meta)
    println(">>> Member $m done  retcode=$(meta.retcode)  wall=$(round(meta.elapsed_sec, digits=1))s")
    flush(stdout)
end

open(joinpath(OUT, "ensemble_meta.txt"), "w") do io
    println(io, "mode=stochastic_from_t0")
    println(io, "t0=0.0")
    println(io, "t_end=", T_END)
    println(io, "ic_seed=", IC_SEED)
    println(io, "n_members=", N_MEMBERS)
    println(io, "step_rel_eps=", STEP_REL_EPS)
    if SAVE_DT !== nothing
        println(io, "save_dt=", SAVE_DT)
    else
        println(io, "save_interval=", SAVE_INTERVAL)
    end
    for (i, m) in enumerate(member_meta)
        println(io, "member_", i, "_seed=", m.perturb_seed)
        println(io, "member_", i, "_elapsed=", round(m.elapsed_sec; digits=1))
    end
end
println("\n>>> Simulation complete: $OUT")
