#!/usr/bin/env julia
"""Smoke test: KH AMR for 10 time steps, save VTU snapshots."""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))

using OrdinaryDiffEqLowStorageRK

const OUT = joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_10steps")
const VTU = joinpath(OUT, "vtu")
const N_STEPS = 10
const SAVE_INTERVAL = 2

mkpath(VTU)

ode, semi, tspan, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(1; save_interval=SAVE_INTERVAL)

save_solution = SaveSolutionCallback(
    interval = SAVE_INTERVAL,
    save_initial_solution = true,
    save_final_solution = true,
    output_directory = VTU,
    solution_variables = cons2prim,
)

callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                        amr_callback, stepsize_callback)

println(">>> Running KH AMR: maxiters=", N_STEPS, "  vtu=", VTU)
flush(stdout)

t0 = time()
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, ode_default_options()..., callback = callbacks, maxiters = N_STEPS)
elapsed = time() - t0

println(">>> DONE retcode=", sol.retcode, "  nsteps=", length(sol.t),
        "  t_final=", round(sol.t[end], digits=6), "  wall=", round(elapsed, digits=1), "s")
println(">>> VTU_DIR=", VTU)
flush(stdout)
