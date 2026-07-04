"""Shared Kelvin-Helmholtz Trixi AMR setup."""

using OrdinaryDiffEqLowStorageRK: DiscreteCallback
using Random
using Trixi
using HDF5

const KH_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

"""Perturbed KH IC: same tanh shear layer, random phase/amplitude in v2 (seed → trajectory)."""
function make_kh_initial_condition(seed::Int)
    rng = MersenneTwister(seed)
    phi = 2π * rand(rng)
    amp_scale = 1.0f0 + 0.05f0 * (rand(rng) - 0.5f0)
    interface_noise = 0.02f0 * (rand(rng) - 0.5f0)

    function initial_condition(x, t, equations::CompressibleEulerEquations2D)
        RealT = eltype(x)
        slope = 15
        B = tanh(slope * x[2] + 7.5f0) - tanh(slope * x[2] - 7.5f0)
        rho = 0.5f0 + 0.75f0 * B + convert(RealT, interface_noise) * B * (1 - B)
        v1 = 0.5f0 * (B - 1)
        v2 = convert(RealT, 0.1 * amp_scale) * sin(2 * π * x[1] + phi)
        p = 1
        return prim2cons(SVector(rho, v1, v2, p), equations)
    end

    return initial_condition
end

function kh_equations_solver()
    gamma = 1.4
    equations = CompressibleEulerEquations2D(gamma)
    surface_flux = FluxLaxFriedrichs(max_abs_speed_naive)
    volume_flux = flux_ranocha
    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)
    indicator_sc = IndicatorHennemannGassner(equations, basis,
                                             alpha_max = 0.002,
                                             alpha_min = 0.0001,
                                             alpha_smooth = true,
                                             variable = density_pressure)
    volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                     volume_flux_dg = volume_flux,
                                                     volume_flux_fv = surface_flux)
    solver = DGSEM(basis, surface_flux, volume_integral)
    return equations, solver
end

function kh_amr_callback(semi, refinement_level::Int, max_amr_level::Int; adapt_ic::Bool = true)
    amr_indicator = IndicatorHennemannGassner(semi,
                                              alpha_max = 1.0,
                                              alpha_min = 0.0001,
                                              alpha_smooth = false,
                                              variable = Trixi.density)
    amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                          base_level = max(4, refinement_level - 1),
                                          med_level = 0, med_threshold = 0.0003,
                                          max_level = max_amr_level, max_threshold = 0.003)
    return AMRCallback(semi, amr_controller,
                       interval = 1,
                       adapt_initial_condition = adapt_ic,
                       adapt_initial_condition_only_refine = adapt_ic)
end

function build_kh_simulation(seed::Int;
                             save_interval::Int = 50,
                             alive_interval::Int = 30,
                             refinement_level::Int = 5,
                             n_cells_max::Int = 100_000,
                             enable_amr::Bool = true,
                             max_amr_level::Int = 6)
    equations, solver = kh_equations_solver()
    initial_condition = make_kh_initial_condition(seed)

    mesh = TreeMesh((-1.0, -1.0), (1.0, 1.0),
                    initial_refinement_level = refinement_level,
                    n_cells_max = n_cells_max, periodicity = true)

    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)

    tspan = (0.0, 3.0)
    ode = semidiscretize(semi, tspan)

    summary_callback = SummaryCallback()
    alive_callback = AliveCallback(analysis_interval = alive_interval)
    stepsize_callback = StepsizeCallback(cfl = 1.3)

    amr_callback = enable_amr ? kh_amr_callback(semi, refinement_level, max_amr_level) : nothing

    return ode, semi, tspan, save_interval, summary_callback, alive_callback,
           stepsize_callback, amr_callback
end

function run_kh_trajectory(traj_id::Int, seed::Int, out_dir::String;
                           save_interval::Int = 50)
    mkpath(out_dir)
    vtu_dir = joinpath(out_dir, "vtu")
    mkpath(vtu_dir)

    ode, semi, tspan, save_interval, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(seed; save_interval=save_interval)

    save_solution = SaveSolutionCallback(interval = save_interval,
                                         save_initial_solution = true,
                                         save_final_solution = true,
                                         output_directory = vtu_dir,
                                         solution_variables = cons2prim)

    callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                            stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                                amr_callback, stepsize_callback)
    end

    t0 = time()
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = 1, ode_default_options()..., callback = callbacks)
    elapsed = time() - t0

    meta = (
        traj_id = traj_id,
        seed = seed,
        tspan = collect(tspan),
        retcode = string(sol.retcode),
        nsteps = length(sol.t),
        elapsed_sec = elapsed,
        vtu_dir = vtu_dir,
        save_interval = save_interval,
    )
    open(joinpath(out_dir, "meta.json"), "w") do io
        println(io, "traj_id=", traj_id)
        println(io, "seed=", seed)
        println(io, "retcode=", sol.retcode)
        println(io, "nsteps=", length(sol.t))
        println(io, "elapsed_sec=", round(elapsed; digits=1))
        println(io, "vtu_dir=", vtu_dir)
    end

    return sol, meta
end

"""Add member-specific random perturbation to state (single shot)."""
function perturb_member!(integrator, seed::Int; rel_eps::Float64 = 3e-4)
    rng = MersenneTwister(seed)
    u = integrator.u
    for i in eachindex(u)
        u[i] += rel_eps * abs(u[i]) * randn(rng)
    end
    return nothing
end

"""Per-step stochastic perturbation (member-specific RNG stream)."""
mutable struct StepPerturbCallback
    rng::MersenneTwister
    rel_eps::Float64
end

function (cb::StepPerturbCallback)(integrator)
    u = integrator.u
    for i in eachindex(u)
        u[i] += cb.rel_eps * abs(u[i]) * randn(cb.rng)
    end
    Trixi.derivative_discontinuity!(integrator, false)
    return nothing
end

function step_perturb_callback(seed::Int; rel_eps::Float64 = 3e-5)
    condition = (u, t, integrator) -> integrator.stats.naccept > 0
    cb = StepPerturbCallback(MersenneTwister(seed), rel_eps)
    return DiscreteCallback(condition, cb, save_positions = (false, false))
end

"""Save by fixed physical Δt (`save_dt`) or every `save_interval` steps (mutually exclusive)."""
function kh_save_solution_callback(vtu_dir::String;
                                   save_dt::Union{Nothing,Float64} = nothing,
                                   save_interval::Int = 40,
                                   save_initial_solution::Bool = true,
                                   save_final_solution::Bool = true)
    if save_dt !== nothing
        return SaveSolutionCallback(
            dt = save_dt + 1.0e-10,
            save_initial_solution = save_initial_solution,
            save_final_solution = save_final_solution,
            output_directory = vtu_dir,
            solution_variables = cons2prim,
        )
    end
    return SaveSolutionCallback(
        interval = save_interval,
        save_initial_solution = save_initial_solution,
        save_final_solution = save_final_solution,
        output_directory = vtu_dir,
        solution_variables = cons2prim,
    )
end

"""Run one ensemble member from t=0 with per-step stochastic perturbations."""
function run_kh_member_stochastic(
    member_id::Int,
    perturb_seed::Int,
    t_end::Float64,
    out_dir::String;
    ic_seed::Int = 1,
    save_interval::Int = 40,
    save_dt::Union{Nothing,Float64} = nothing,
    step_rel_eps::Float64 = 3e-5,
    refinement_level::Int = 5,
    n_cells_max::Int = 100_000,
    enable_amr::Bool = true,
    max_amr_level::Int = 6,
)
    mkpath(out_dir)
    vtu_dir = joinpath(out_dir, "vtu")
    mkpath(vtu_dir)

    ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(
        ic_seed;
        save_interval = save_interval,
        refinement_level = refinement_level,
        n_cells_max = n_cells_max,
        enable_amr = enable_amr,
        max_amr_level = max_amr_level,
    )
    ode = remake(ode; tspan = (0.0, t_end))

    save_solution = kh_save_solution_callback(
        vtu_dir;
        save_dt = save_dt,
        save_interval = save_interval,
        save_initial_solution = true,
        save_final_solution = true,
    )
    step_perturb = step_perturb_callback(perturb_seed; rel_eps = step_rel_eps)

    callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                            step_perturb, stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                                step_perturb, amr_callback, stepsize_callback)
    end

    wall = time()
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = 1, ode_default_options()..., callback = callbacks)
    elapsed = time() - wall

    meta = (
        member_id = member_id,
        ic_seed = ic_seed,
        perturb_seed = perturb_seed,
        t0 = 0.0,
        t_end = sol.t[end],
        step_rel_eps = step_rel_eps,
        save_dt = save_dt,
        save_interval = save_dt === nothing ? save_interval : 0,
        retcode = string(sol.retcode),
        elapsed_sec = elapsed,
        vtu_dir = vtu_dir,
    )
    open(joinpath(out_dir, "meta.json"), "w") do io
        for (k, v) in pairs(meta)
            println(io, k, "=", v)
        end
    end
    return sol, meta
end

function latest_restart_file(restart_dir::String)
    files = filter(f -> startswith(f, "restart_") && endswith(f, ".h5"), readdir(restart_dir))
    isempty(files) && error("No restart files in $restart_dir")
    return joinpath(restart_dir, sort(files)[end])
end

"""Run reference to t_stop; return final solution H5 path and physical time."""
function run_kh_to_snapshot(t_stop::Float64, out_dir::String;
                            seed::Int = 1,
                            refinement_level::Int = 5,
                            n_cells_max::Int = 100_000,
                            enable_amr::Bool = true)
    mkpath(out_dir)
    vtu_dir = joinpath(out_dir, "vtu")
    mkpath(vtu_dir)

    ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(
        seed;
        refinement_level = refinement_level,
        n_cells_max = n_cells_max,
        enable_amr = enable_amr,
    )
    ode = remake(ode; tspan = (0.0, t_stop))

    save_solution = SaveSolutionCallback(
        interval = 10_000,
        save_initial_solution = false,
        save_final_solution = true,
        output_directory = vtu_dir,
        solution_variables = cons2prim,
    )
    callbacks = CallbackSet(summary_callback, alive_callback, save_solution, stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                                amr_callback, stepsize_callback)
    end

    t0 = time()
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = 1, ode_default_options()..., callback = callbacks)
    elapsed = time() - t0

    files = filter(f -> startswith(f, "solution_") && endswith(f, ".h5"), readdir(vtu_dir))
    isempty(files) && error("No solution saved in $vtu_dir")
    solution_file = joinpath(vtu_dir, sort(files)[end])
    t_final = sol.t[end]
    @info "Snapshot at t=$(round(t_final, digits=4))" solution_file elapsed
    return solution_file, t_final, elapsed
end

"""Run reference trajectory to t_stop and write restart file (seed=1 matches GIF run)."""
function run_kh_to_restart(t_stop::Float64, restart_dir::String;
                           seed::Int = 1,
                           refinement_level::Int = 5,
                           n_cells_max::Int = 100_000,
                           enable_amr::Bool = true)
    mkpath(restart_dir)
    ode, semi, _, _, summary_callback, alive_callback,
    stepsize_callback, amr_callback = build_kh_simulation(
        seed;
        refinement_level = refinement_level,
        n_cells_max = n_cells_max,
        enable_amr = enable_amr,
    )
    ode = remake(ode; tspan = (0.0, t_stop))

    save_restart = SaveRestartCallback(
        interval = 10_000,
        save_final_restart = true,
        output_directory = restart_dir,
    )
    callbacks = CallbackSet(summary_callback, alive_callback, save_restart, stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_restart,
                                amr_callback, stepsize_callback)
    end

    t0 = time()
    sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                dt = 1, ode_default_options()..., callback = callbacks)
    elapsed = time() - t0
    restart_file = latest_restart_file(restart_dir)
    @info "Restart at t=$(round(sol.t[end], digits=4))" restart_file elapsed
    return restart_file, sol, elapsed
end

"""Load primitive snapshot (cons2prim) into integrator state."""
function load_prim_solution_into_integrator!(integrator, solution_file::String, semi)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)
    u = Trixi.wrap_array_native(integrator.u, mesh, equations, solver, cache)
    ni, nj, ne = size(u, 2), size(u, 3), size(u, 4)
    HDF5.h5open(solution_file, "r") do file
        rho = reshape(read(file["variables_1"]), ni, nj, ne)
        v1 = reshape(read(file["variables_2"]), ni, nj, ne)
        v2 = reshape(read(file["variables_3"]), ni, nj, ne)
        p = reshape(read(file["variables_4"]), ni, nj, ne)
        for e in 1:ne, j in 1:nj, i in 1:ni
            cons = prim2cons(SVector(rho[i, j, e], v1[i, j, e], v2[i, j, e], p[i, j, e]), equations)
            for v in eachvariable(equations)
                u[v, i, j, e] = cons[v]
            end
        end
    end
    return nothing
end

function solution_mesh_file(solution_file::String)
    h5_dir = dirname(solution_file)
    step = splitext(split(basename(solution_file), "_")[end])[1]
    mesh_file = joinpath(h5_dir, "mesh_$(step).h5")
    isfile(mesh_file) || error("Missing mesh file $mesh_file for snapshot")
    return mesh_file
end

function load_snapshot_mesh(mesh_file::String; n_cells_max::Int = 100_000)
    ndims, mesh_type, capacity = HDF5.h5open(mesh_file, "r") do file
        read(attributes(file)["ndims"]),
        read(attributes(file)["mesh_type"]),
        read(attributes(file)["capacity"])
    end
    mesh_type == "TreeMesh" || error("Unsupported mesh type $mesh_type")
    initial_capacity = max(n_cells_max, capacity)
    mesh = TreeMesh(Trixi.SerialTree{ndims, Float64}, initial_capacity, RealT = Float64)
    Trixi.load_mesh!(mesh, mesh_file)
    return mesh
end

function snapshot_step(solution_file::String)
    base = basename(solution_file)
    parse(Int, splitext(split(base, "_")[end])[1])
end

"""Branch one member from an existing solution snapshot (same field for all members)."""
function run_kh_member_from_solution(
    solution_file::String,
    member_id::Int,
    perturb_seed::Int,
    t_forward::Float64,
    out_dir::String;
    save_interval::Int = 40,
    rel_eps::Float64 = 3e-4,
    refinement_level::Int = 5,
    n_cells_max::Int = 100_000,
    enable_amr::Bool = true,
    max_amr_level::Int = 6,
)
    mkpath(out_dir)
    vtu_dir = joinpath(out_dir, "vtu")
    mkpath(vtu_dir)

    source_step = snapshot_step(solution_file)
    mesh_file = solution_mesh_file(solution_file)
    mesh = load_snapshot_mesh(mesh_file; n_cells_max = n_cells_max)
    equations, solver = kh_equations_solver()
    ic = make_kh_initial_condition(1)  # unused; state loaded from snapshot
    semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver;
                                        boundary_conditions = boundary_condition_periodic)

    t0 = load_time(solution_file)
    tspan = (t0, t0 + t_forward)
    ode = semidiscretize(semi, tspan)

    save_solution_disc = SaveSolutionCallback(
        interval = save_interval,
        save_initial_solution = false,
        save_final_solution = true,
        output_directory = vtu_dir,
        solution_variables = cons2prim,
    )
    summary_callback = SummaryCallback()
    alive_callback = AliveCallback(analysis_interval = 30)
    stepsize_callback = StepsizeCallback(cfl = 1.3)
    amr_callback = enable_amr ? kh_amr_callback(semi, refinement_level, max_amr_level; adapt_ic = false) : nothing

    callbacks = CallbackSet(summary_callback, alive_callback, save_solution_disc, stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_solution_disc,
                                amr_callback, stepsize_callback)
    end

    integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false);
                      dt = 1, ode_default_options()..., callback = callbacks)
    load_prim_solution_into_integrator!(integrator, solution_file, semi)
    integrator.t = t0
    integrator.iter = source_step
    integrator.stats.naccept = source_step
    perturb_member!(integrator, perturb_seed; rel_eps = rel_eps)

    mesh_, _, _, _ = Trixi.mesh_equations_solver_cache(semi)
    mesh_.unsaved_changes = true
    save_solution_disc.affect!(integrator)

    wall = time()
    sol = solve!(integrator)
    elapsed = time() - wall

    meta = (
        member_id = member_id,
        perturb_seed = perturb_seed,
        source_file = solution_file,
        source_step = source_step,
        t0 = t0,
        t_end = sol.t[end],
        rel_eps = rel_eps,
        retcode = string(sol.retcode),
        elapsed_sec = elapsed,
        vtu_dir = vtu_dir,
    )
    open(joinpath(out_dir, "meta.json"), "w") do io
        for (k, v) in pairs(meta)
            println(io, k, "=", v)
        end
    end
    return sol, meta
end

"""Branch from snapshot; per-step stochastic noise (no initial perturb at branch)."""
function run_kh_member_from_solution_stochastic(
    solution_file::String,
    member_id::Int,
    perturb_seed::Int,
    t_end::Float64,
    out_dir::String;
    save_interval::Int = 40,
    save_dt::Union{Nothing,Float64} = nothing,
    step_rel_eps::Float64 = 3e-4,
    refinement_level::Int = 5,
    n_cells_max::Int = 100_000,
    enable_amr::Bool = true,
    max_amr_level::Int = 6,
)
    mkpath(out_dir)
    vtu_dir = joinpath(out_dir, "vtu")
    mkpath(vtu_dir)

    source_step = snapshot_step(solution_file)
    mesh_file = solution_mesh_file(solution_file)
    mesh = load_snapshot_mesh(mesh_file; n_cells_max = n_cells_max)
    equations, solver = kh_equations_solver()
    ic = make_kh_initial_condition(1)
    semi = SemidiscretizationHyperbolic(mesh, equations, ic, solver;
                                        boundary_conditions = boundary_condition_periodic)

    t0 = load_time(solution_file)
    tspan = (t0, t_end)
    ode = semidiscretize(semi, tspan)

    save_solution_disc = kh_save_solution_callback(
        vtu_dir;
        save_dt = save_dt,
        save_interval = save_interval,
        save_initial_solution = false,
        save_final_solution = true,
    )
    step_perturb = step_perturb_callback(perturb_seed; rel_eps = step_rel_eps)
    summary_callback = SummaryCallback()
    alive_callback = AliveCallback(analysis_interval = 30)
    stepsize_callback = StepsizeCallback(cfl = 1.3)
    amr_callback = enable_amr ? kh_amr_callback(semi, refinement_level, max_amr_level; adapt_ic = false) : nothing

    callbacks = CallbackSet(summary_callback, alive_callback, save_solution_disc,
                            step_perturb, stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_solution_disc,
                                step_perturb, amr_callback, stepsize_callback)
    end

    integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false);
                      dt = 1, ode_default_options()..., callback = callbacks)
    load_prim_solution_into_integrator!(integrator, solution_file, semi)
    integrator.t = t0
    integrator.iter = source_step
    integrator.stats.naccept = source_step

    mesh_, _, _, _ = Trixi.mesh_equations_solver_cache(semi)
    mesh_.unsaved_changes = true
    save_solution_disc.affect!(integrator)

    wall = time()
    sol = solve!(integrator)
    elapsed = time() - wall

    meta = (
        member_id = member_id,
        perturb_seed = perturb_seed,
        source_file = solution_file,
        source_step = source_step,
        t0 = t0,
        t_end = sol.t[end],
        step_rel_eps = step_rel_eps,
        save_dt = save_dt,
        save_interval = save_dt === nothing ? save_interval : 0,
        retcode = string(sol.retcode),
        elapsed_sec = elapsed,
        vtu_dir = vtu_dir,
    )
    open(joinpath(out_dir, "meta.json"), "w") do io
        for (k, v) in pairs(meta)
            println(io, k, "=", v)
        end
    end
    return sol, meta
end

"""Branch one member from shared restart: perturb then integrate forward."""
function run_kh_member_from_restart(
    restart_file::String,
    member_id::Int,
    perturb_seed::Int,
    t_end::Float64,
    out_dir::String;
    save_interval::Int = 40,
    rel_eps::Float64 = 3e-4,
    refinement_level::Int = 5,
    n_cells_max::Int = 100_000,
    enable_amr::Bool = true,
    max_amr_level::Int = 6,
)
    mkpath(out_dir)
    vtu_dir = joinpath(out_dir, "vtu")
    mkpath(vtu_dir)

    equations, solver = kh_equations_solver()
    mesh = load_mesh(restart_file; n_cells_max = n_cells_max)
    # IC unused — state loaded from restart
    dummy_ic = make_kh_initial_condition(1)
    semi = SemidiscretizationHyperbolic(mesh, equations, dummy_ic, solver;
                                        boundary_conditions = boundary_condition_periodic)

    t0 = load_time(restart_file)
    tspan = (t0, t_end)
    dt = load_dt(restart_file)
    ode = semidiscretize(semi, tspan, restart_file)

    save_solution = SaveSolutionCallback(
        interval = save_interval,
        save_initial_solution = true,
        save_final_solution = true,
        output_directory = vtu_dir,
        solution_variables = cons2prim,
    )
    summary_callback = SummaryCallback()
    alive_callback = AliveCallback(analysis_interval = 30)
    stepsize_callback = StepsizeCallback(cfl = 1.3)
    amr_callback = enable_amr ? kh_amr_callback(semi, refinement_level, max_amr_level; adapt_ic = false) : nothing

    callbacks = CallbackSet(summary_callback, alive_callback, save_solution, stepsize_callback)
    if amr_callback !== nothing
        callbacks = CallbackSet(summary_callback, alive_callback, save_solution,
                                amr_callback, stepsize_callback)
    end

    integrator = init(ode, CarpenterKennedy2N54(williamson_condition = false);
                      dt = dt, ode_default_options()..., callback = callbacks)
    load_timestep!(integrator, restart_file)
    perturb_member!(integrator, perturb_seed; rel_eps = rel_eps)

    wall = time()
    sol = solve!(integrator)
    elapsed = time() - wall

    meta = (
        member_id = member_id,
        perturb_seed = perturb_seed,
        t0 = t0,
        t_end = sol.t[end],
        rel_eps = rel_eps,
        retcode = string(sol.retcode),
        elapsed_sec = elapsed,
        vtu_dir = vtu_dir,
    )
    open(joinpath(out_dir, "meta.json"), "w") do io
        for (k, v) in pairs(meta)
            println(io, k, "=", v)
        end
    end
    return sol, meta
end
