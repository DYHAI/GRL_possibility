#!/usr/bin/env julia
"""
1024×1024 uniform Trixi KH (refinement_level=10 → 2^10 cells per side).

Same physics as elixir: DGSEM p=3, Lax–Friedrichs, Ranocha + shock capturing.
AMR disabled for a fixed 1024×1024 base grid.
"""

using Pkg
Pkg.activate(@__DIR__)

include(joinpath(@__DIR__, "kh_common.jl"))
using OrdinaryDiffEqLowStorageRK

const REF_LEVEL = 10          # 2^10 = 1024 cells / side
const N_CELLS = 1024 * 1024
const T_END = parse(Float64, get(ENV, "KH_T_END", "3.0"))
const SAVE_INTERVAL = parse(Int, get(ENV, "KH_SAVE_INTERVAL", "300"))
const OUT = joinpath(KH_ROOT, "outputs", "kelvin_helmholtz", "trixi_1024")
const VTU = joinpath(OUT, "vtu")

mkpath(VTU)

ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(
        1;
        save_interval = SAVE_INTERVAL,
        refinement_level = REF_LEVEL,
        n_cells_max = N_CELLS + 100_000,
        enable_amr = false,
    )

ode = remake(ode; tspan = (0.0, T_END))

save_solution = SaveSolutionCallback(
    interval = SAVE_INTERVAL,
    save_initial_solution = true,
    save_final_solution = true,
    output_directory = VTU,
    solution_variables = cons2prim,
)

callbacks = CallbackSet(summary_callback, alive_callback, save_solution, stepsize_callback)

ncells = 2^REF_LEVEL
println(">>> KH 1024×1024  cells=", ncells, "×", ncells, "  t=0..", T_END)
println(">>> save_interval=", SAVE_INTERVAL, "  OUT=", OUT)
println(">>> threads=", Threads.nthreads())
flush(stdout)

t0 = time()
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1, ode_default_options()..., callback = callbacks)
elapsed = time() - t0

open(joinpath(OUT, "meta.json"), "w") do io
    println(io, "refinement_level=", REF_LEVEL)
    println(io, "ncells_per_side=", ncells)
    println(io, "t_end=", T_END)
    println(io, "retcode=", sol.retcode)
    println(io, "nsteps=", length(sol.t))
    println(io, "elapsed_sec=", round(elapsed; digits=1))
end

println(">>> DONE retcode=", sol.retcode, "  t_final=", round(sol.t[end], digits=4),
        "  wall=", round(elapsed / 60; digits=1), " min")
println(">>> Next: NVIS=1 julia export_npz.jl ", OUT)
println(">>>       python render_trixi_hd.py --data-dir ", OUT, " --grid-n 1024")
flush(stdout)
