#!/usr/bin/env julia
"""
Fast KH preview: ONE trajectory, ONE snapshot, short time.

Default: Trixi elixir mesh (32 + AMR), t=0→1, save final only.
Env: KH_REF KH_T_END KH_AMR (0/1)
"""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const REF = parse(Int, get(ENV, "KH_REF", "9"))       # 9 → 512×512 uniform
const T_END = parse(Float64, get(ENV, "KH_T_END", "0.5"))
const USE_AMR = get(ENV, "KH_AMR", "0") != "0"
const OUT = get(ENV, "KH_OUT", joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_512"))
const VTU = joinpath(OUT, "vtu")

rm(OUT; recursive=true, force=true)
mkpath(VTU)

n_cells_max = USE_AMR ? 100_000 : (2^(2 * REF) + 50_000)

ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(
        1;
        save_interval = 999_999,
        alive_interval = 30,
        refinement_level = REF,
        n_cells_max = n_cells_max,
        enable_amr = USE_AMR,
    )

ode = remake(ode; tspan = (0.0, T_END))

save_solution = SaveSolutionCallback(
    interval = 999_999,
    save_initial_solution = false,
    save_final_solution = true,
    output_directory = VTU,
    solution_variables = cons2prim,
)

callbacks = CallbackSet(summary_callback, alive_callback, save_solution, stepsize_callback)
if amr_callback !== nothing
    callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                            amr_callback, stepsize_callback)
end

println(">>> PREVIEW  ref=", REF, "  AMR=", USE_AMR, "  t=0..", T_END)
println(">>> OUT=", OUT, "  threads=", Threads.nthreads())
flush(stdout)

t0 = time()
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, ode_default_options()..., callback = callbacks)
elapsed = time() - t0

open(joinpath(OUT, "meta.json"), "w") do io
    println(io, "refinement_level=", REF)
    println(io, "amr=", USE_AMR)
    println(io, "t_end=", T_END)
    println(io, "retcode=", sol.retcode)
    println(io, "elapsed_sec=", round(elapsed; digits=1))
end

println(">>> DONE  t=", round(sol.t[end], digits=4), "  wall=", round(elapsed, digits=1), "s")
println(">>> Render: bash run_preview_figure.sh")
flush(stdout)
