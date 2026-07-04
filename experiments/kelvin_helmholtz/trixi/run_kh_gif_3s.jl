#!/usr/bin/env julia
"""
KH t=0→3 with sampled snapshots for GIF (Trixi elixir: 32 + AMR).
"""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const REF = parse(Int, get(ENV, "KH_REF", "5"))
const T_END = parse(Float64, get(ENV, "KH_T_END", "3.0"))
const USE_AMR = get(ENV, "KH_AMR", "1") != "0"
const SAVE_INTERVAL = parse(Int, get(ENV, "KH_SAVE_INTERVAL", "40"))
const OUT = get(ENV, "KH_OUT", joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_gif_3s"))
const VTU = joinpath(OUT, "vtu")

mkpath(OUT)
for sub in ("vtu", "vtu_converted", "frames", "hd")
    rm(joinpath(OUT, sub); recursive=true, force=true)
end
for f in ("snapshots.npz", "meta.json", "gif_summary.json")
    rm(joinpath(OUT, f); force=true)
end
mkpath(VTU)

n_cells_max = USE_AMR ? 100_000 : (2^(2 * REF) + 50_000)

ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(
        1;
        save_interval = SAVE_INTERVAL,
        alive_interval = 30,
        refinement_level = REF,
        n_cells_max = n_cells_max,
        enable_amr = USE_AMR,
    )

ode = remake(ode; tspan = (0.0, T_END))

save_solution = SaveSolutionCallback(
    interval = SAVE_INTERVAL,
    save_initial_solution = true,
    save_final_solution = true,
    output_directory = VTU,
    solution_variables = cons2prim,
)
save_restart = SaveRestartCallback(
    interval = SAVE_INTERVAL,
    save_final_restart = true,
    output_directory = joinpath(OUT, "restart"),
)

callbacks = CallbackSet(summary_callback, alive_callback, save_solution, save_restart, stepsize_callback)
if amr_callback !== nothing
    callbacks = CallbackSet(summary_callback, alive_callback, save_solution, save_restart,
                            amr_callback, stepsize_callback)
end

println(">>> GIF run  ref=", REF, "  AMR=", USE_AMR, "  t=0..", T_END)
println(">>> save_interval=", SAVE_INTERVAL, "  OUT=", OUT)
println(">>> threads=", Threads.nthreads())
flush(stdout)

t0 = time()
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, ode_default_options()..., callback = callbacks)
elapsed = time() - t0

open(joinpath(OUT, "meta.json"), "w") do io
    println(io, "refinement_level=", REF)
    println(io, "amr=", USE_AMR)
    println(io, "t_end=", T_END)
    println(io, "save_interval=", SAVE_INTERVAL)
    println(io, "retcode=", sol.retcode)
    println(io, "elapsed_sec=", round(elapsed; digits=1))
end

println(">>> DONE  t=", round(sol.t[end], digits=4), "  wall=", round(elapsed, digits=1), "s")
flush(stdout)
