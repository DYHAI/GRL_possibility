#!/usr/bin/env julia
"""Install local Trixi + deps for the KH AMR experiment."""

using Pkg

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const TRIXI_SRC = joinpath(ROOT, "experiments", "Trixi.jl-main")
const ENV_DIR = @__DIR__

println(">>> JULIA_PKG_SERVER = ", get(ENV, "JULIA_PKG_SERVER", "default"))
flush(stdout)

@info "Activating Julia project" ENV_DIR
Pkg.activate(ENV_DIR)

@info "Developing local Trixi from" TRIXI_SRC
Pkg.develop(path=TRIXI_SRC)

@info "Adding OrdinaryDiffEqLowStorageRK"
Pkg.add("OrdinaryDiffEqLowStorageRK")

@info "Precompiling (may take several minutes)..."
Pkg.precompile()

@info "Setup complete."
println(">>> SETUP_COMPLETE")
flush(stdout)
