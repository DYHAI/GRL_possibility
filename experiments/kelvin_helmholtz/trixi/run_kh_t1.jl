#!/usr/bin/env julia
"""Run KH AMR to t=1.0 (Trixi elixir settings), save HDF5 snapshots."""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const T_END = parse(Float64, get(ENV, "KH_T_END", "1.0"))
const SAVE_INTERVAL = parse(Int, get(ENV, "KH_SAVE_INTERVAL", "50"))
const OUT = joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_t$(replace(string(T_END), '.' => 'p'))")
const VTU = joinpath(OUT, "vtu")

mkpath(VTU)

ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(1; save_interval=SAVE_INTERVAL)

ode = remake(ode; tspan=(0.0, T_END))

save_solution = SaveSolutionCallback(
    interval = SAVE_INTERVAL,
    save_initial_solution = true,
    save_final_solution = true,
    output_directory = VTU,
    solution_variables = cons2prim,
)

callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                        amr_callback, stepsize_callback)

println(">>> KH AMR  t=0..", T_END, "  save_interval=", SAVE_INTERVAL)
println(">>> OUT=", OUT)
flush(stdout)

t0 = time()
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, ode_default_options()..., callback = callbacks)
elapsed = time() - t0

println(">>> DONE retcode=", sol.retcode, "  t_final=", round(sol.t[end], digits=4),
        "  wall=", round(elapsed, digits=1), "s")
println(">>> VTU_DIR=", VTU)
flush(stdout)
