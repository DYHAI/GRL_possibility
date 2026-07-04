#!/usr/bin/env julia
"""
Branch ensemble from snapshot:
  1) Deterministic reference t=0..T_BRANCH (no noise)
  2) N members from that snapshot, per-step noise to T_END
"""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const REF = parse(Int, get(ENV, "KH_REF", "5"))
const USE_AMR = get(ENV, "KH_AMR", "1") != "0"
const T_BRANCH = parse(Float64, get(ENV, "KH_T_BRANCH", "2.02"))
const T_END = parse(Float64, get(ENV, "KH_T_END", "7.0"))
const IC_SEED = parse(Int, get(ENV, "KH_IC_SEED", "1"))
const N_MEMBERS = parse(Int, get(ENV, "KH_N_MEMBERS", "5"))
const SAVE_INTERVAL = parse(Int, get(ENV, "KH_SAVE_INTERVAL", "40"))
const SAVE_DT = let s = get(ENV, "KH_SAVE_DT", "0.1"); isempty(strip(s)) ? nothing : parse(Float64, s) end
const STEP_REL_EPS = parse(Float64, get(ENV, "KH_STEP_REL_EPS", "3e-4"))
const OUT = get(ENV, "KH_OUT", joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_ensemble_branch_2p02_7s"))

n_cells_max = USE_AMR ? 100_000 : (2^(2 * REF) + 50_000)

mkpath(OUT)
for sub in ("member_001", "member_002", "member_003", "member_004", "member_005", "frames")
    rm(joinpath(OUT, sub); recursive = true, force = true)
end
for f in ("ensemble_snapshots.npz", "ensemble_fields_4var.npz", "unet_training_512.npz",
          "ensemble_meta.txt", "gif_summary.json", "kh_ensemble_branch.gif", "run.log")
    rm(joinpath(OUT, f); force = true)
end
rm(joinpath(OUT, "baseline"); recursive = true, force = true)

println(">>> Phase 1/2: deterministic reference t=0..$(T_BRANCH) (seed=$IC_SEED, no noise)")
flush(stdout)
baseline_dir = joinpath(OUT, "baseline")
solution_file, t0, elapsed_ref = run_kh_to_snapshot(
    T_BRANCH, baseline_dir;
    seed = IC_SEED,
    refinement_level = REF,
    n_cells_max = n_cells_max,
    enable_amr = USE_AMR,
)
println(">>> Baseline done  t=$(round(t0, digits=4))  wall=$(round(elapsed_ref, digits=1))s")
println(">>>   snapshot: $solution_file")
flush(stdout)

println("\n>>> Phase 2/2: $N_MEMBERS members  t=$(round(t0, digits=4))..$(T_END)  step_rel_eps=$STEP_REL_EPS")
if SAVE_DT !== nothing
    println(">>>   save_dt=$(SAVE_DT) s (fixed physical time output)")
else
    println(">>>   save_interval=$(SAVE_INTERVAL) steps")
end
flush(stdout)

member_meta = []
for m in 1:N_MEMBERS
    perturb_seed = 1000 + m
    member_dir = joinpath(OUT, "member_$(lpad(m, 3, '0'))")
    println("\n>>> Member $m  perturb_seed=$perturb_seed")
    flush(stdout)
    _, meta = run_kh_member_from_solution_stochastic(
        solution_file, m, perturb_seed, T_END, member_dir;
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
    println(io, "mode=branch_stochastic")
    println(io, "source_file=", solution_file)
    println(io, "t_branch=", T_BRANCH)
    println(io, "t0=", t0)
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
