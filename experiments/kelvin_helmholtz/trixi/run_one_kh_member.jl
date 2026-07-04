#!/usr/bin/env julia
"""Run a single stochastic ensemble member (for incremental large runs)."""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const REF = parse(Int, get(ENV, "KH_REF", "5"))
const USE_AMR = get(ENV, "KH_AMR", "1") != "0"
const T_END = parse(Float64, get(ENV, "KH_T_END", "5.0"))
const IC_SEED = parse(Int, get(ENV, "KH_IC_SEED", "1"))
const MEMBER_ID = parse(Int, get(ENV, "KH_MEMBER_ID", "1"))
const SAVE_INTERVAL = parse(Int, get(ENV, "KH_SAVE_INTERVAL", "40"))
const SAVE_DT = let s = get(ENV, "KH_SAVE_DT", "0.2"); isempty(strip(s)) ? nothing : parse(Float64, s) end
const STEP_REL_EPS = parse(Float64, get(ENV, "KH_STEP_REL_EPS", "3e-4"))
const OUT = get(ENV, "KH_OUT", joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_ensemble_200"))

n_cells_max = USE_AMR ? 100_000 : (2^(2 * REF) + 50_000)

mkpath(OUT)
m = MEMBER_ID
perturb_seed = 1000 + m
member_dir = joinpath(OUT, "member_$(lpad(m, 3, '0'))")

println(">>> Single member $m  t=0..$(T_END)  perturb_seed=$perturb_seed  step_rel_eps=$STEP_REL_EPS")
if SAVE_DT !== nothing
    println(">>>   save_dt=$(SAVE_DT) s")
else
    println(">>>   save_interval=$(SAVE_INTERVAL) steps")
end
println(">>>   OUT=$member_dir")
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
println(">>> Member $m done  retcode=$(meta.retcode)  wall=$(round(meta.elapsed_sec, digits=1))s")
flush(stdout)

meta_path = joinpath(OUT, "ensemble_meta.txt")
if !isfile(meta_path)
    open(meta_path, "w") do io
        println(io, "mode=stochastic_from_t0")
        println(io, "t0=0.0")
        println(io, "t_end=", T_END)
        println(io, "ic_seed=", IC_SEED)
        println(io, "step_rel_eps=", STEP_REL_EPS)
        if SAVE_DT !== nothing
            println(io, "save_dt=", SAVE_DT)
        else
            println(io, "save_interval=", SAVE_INTERVAL)
        end
    end
end
open(meta_path, "a") do io
    println(io, "member_", m, "_seed=", meta.perturb_seed)
    println(io, "member_", m, "_elapsed=", round(meta.elapsed_sec; digits=1))
    println(io, "member_", m, "_retcode=", meta.retcode)
end
