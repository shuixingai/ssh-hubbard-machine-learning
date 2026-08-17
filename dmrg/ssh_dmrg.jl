"""
SSH-Hubbard DMRG — Dataset Generator (ITensor Julia)
====================================================
Generates a dataset of SSH-Hubbard ground states using DMRG,
extracting:
  - Ground-state energy
  - Entanglement spectrum (Schmidt values at the middle cut)
  - von Neumann entanglement entropy
  - Charge gap (optional, via 3 independent DMRG runs)

Output: JLD2 file with full parameter grid.

Usage:
  julia ssh_dmrg.jl                        # default parameters
  julia ssh_dmrg.jl --L 40 --grid 15 15 7  # custom grid

The output .jld2 file can be read from Python or Julia for post-processing.
"""

using ITensors
using ITensorMPS
using JLD2
using LinearAlgebra
using Random
using Dates

# ══════════════════════════════════════════════════════════════════════
#  1. SSH-Hubbard Hamiltonian builder
# ══════════════════════════════════════════════════════════════════════

function build_ssh_hubbard_hamiltonian(sites::Vector{<:Index},
                                        t1::Float64, t2::Float64,
                                        U::Float64, V::Float64=0.0)
    """Build OpSum for SSH-Hubbard model at half-filling.

    V is the nearest-neighbour Coulomb repulsion V Σ n_i n_{i+1}.
    """
    N = length(sites)
    os = OpSum()

    # Alternating hopping
    for j in 1:(N - 1)
        t = isodd(j) ? t1 : t2
        os += -t, "Cdagup", j, "Cup", j + 1
        os += -t, "Cdagup", j + 1, "Cup", j
        os += -t, "Cdagdn", j, "Cdn", j + 1
        os += -t, "Cdagdn", j + 1, "Cdn", j
    end

    # Hubbard interaction
    for j in 1:N
        os += U, "Nupdn", j
    end

    # Nearest-neighbour Coulomb repulsion: V n_i n_{i+1}
    if V != 0.0
        for j in 1:(N - 1)
            os += V, "Ntot", j, "Ntot", j + 1
        end
    end

    return os
end


# ══════════════════════════════════════════════════════════════════════
#  2. DMRG solver
# ══════════════════════════════════════════════════════════════════════

function run_dmrg_for_sector(sites::Vector{<:Index}, t1::Float64,
                              t2::Float64, U::Float64; V::Float64=0.0,
                              n_up=nothing, n_dn=nothing,
                              maxdim=[20, 60, 100, 200, 400, 600, 800, 1000],
                              nsweeps=8, cutoff=1e-8, noise_scale=1e-6,
                              outputlevel=0, seed=42)
    """
    Run DMRG for SSH-Hubbard model in a given particle-number sector.

    If n_up / n_dn are given, a product-state initial MPS in that sector
    is constructed.  Otherwise half-filling is assumed.
    """
    N = length(sites)
    if n_up === nothing
        n_up = N ÷ 2
    end
    if n_dn === nothing
        n_dn = N ÷ 2
    end

    # Random seed for reproducibility
    Random.seed!(seed)

    # Build Hamiltonian (with optional nearest-neighbour V)
    os = build_ssh_hubbard_hamiltonian(sites, t1, t2, U, V)
    H = MPO(os, sites)

    # Initial product state at half-filling
    state = fill("Emp", N)
    filled_up = 0
    filled_dn = 0
    for j in 1:N
        if filled_up < n_up && filled_dn < n_dn
            if isodd(j)
                state[j] = "Up"
                filled_up += 1
            else
                state[j] = "Dn"
                filled_dn += 1
            end
        elseif filled_up < n_up
            state[j] = "Up"
            filled_up += 1
        elseif filled_dn < n_dn
            state[j] = "Dn"
            filled_dn += 1
        end
    end

    psi0 = randomMPS(sites, state, min(10, min(n_up + n_dn, 2N - n_up - n_dn) + 1))

    # Noise schedule
    noise = [noise_scale * (0.1)^(s - 1) for s in 1:nsweeps]

    # DMRG
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff=cutoff,
                        noise=noise, outputlevel=outputlevel)

    return energy, psi
end


# ══════════════════════════════════════════════════════════════════════
#  3. Observables from the MPS
# ══════════════════════════════════════════════════════════════════════

function extract_entanglement(psi::MPS, n_max::Int=20)
    """
    From an MPS, extract:
      - Schmidt values λ_α at the middle bond
      - Entanglement energies ε_α = -ln(λ²_α)
      - von Neumann entanglement entropy S_vN
    """
    N = length(psi)
    b = N ÷ 2  # bond between site b and b+1

    orthogonalize!(psi, b + 1)

    # After orthogonalize!, the left link of site b+1 (= the bond between
    # b and b+1) stores the Schmidt values
    li = linkind(psi, b)  # link index at bond b
    U_mat, S_diag, V_mat = svd(psi[b + 1], li)
    lambdas = Vector(diag(S_diag))

    # Filter out numerical zeros
    lambdas = lambdas[lambdas .> 1e-14]

    # Schmidt values squared (normalized)
    lambda_sq = Vector(lambdas .^ 2)
    lambda_sq = lambda_sq ./ sum(lambda_sq)

    # von Neumann entropy
    SvN = -sum(lambda_sq .* log.(max.(lambda_sq, 1e-30)))

    # Entanglement spectrum ε_α = -ln(λ²_α), sorted ascending
    epsilon = Vector(-log.(max.(lambda_sq, 1e-30)))
    sort!(epsilon)

    # Pad / truncate to n_max
    n = min(length(epsilon), n_max)
    out_ent = zeros(n_max)
    out_ent[1:n] = epsilon[1:n]

    return out_ent, SvN
end


# ══════════════════════════════════════════════════════════════════════
#  3b. δn and δB observables from MPS
# ══════════════════════════════════════════════════════════════════════

function extract_staggered_charge(psi::MPS, sites::Vector{<:Index})
    """Staggered charge density  δn = 1/L Σ_i (-1)^i ⟨n_i⟩."""
    N = length(psi)
    n_all = expect(psi, "Ntot")  # vector of site occupations
    total = 0.0
    for i in 1:N
        total += (-1.0)^i * n_all[i]
    end
    return total / N
end


function extract_bond_alternation(psi::MPS, sites::Vector{<:Index})
    """Bond-order alternation  δB = 1/(L-1) Σ_i (-1)^i B_i.

    B_i = ⟨c†_{i,↑} c_{i+1,↑} + c†_{i+1,↑} c_{i,↑}
        + c†_{i,↓} c_{i+1,↓} + c†_{i+1,↓} c_{i,↓}⟩
    """
    N = length(psi)
    total = 0.0
    for i in 1:(N - 1)
        os = OpSum()
        os += 1.0, "Cdagup", i, "Cup", i + 1
        os += 1.0, "Cdagup", i + 1, "Cup", i
        os += 1.0, "Cdagdn", i, "Cdn", i + 1
        os += 1.0, "Cdagdn", i + 1, "Cdn", i
        bond_op = MPO(os, sites)
        B_i = real(inner(psi', bond_op, psi))
        total += (-1.0)^i * B_i
    end
    return total / (N - 1)
end


function extract_double_occupancy(psi::MPS, sites::Vector{<:Index})
    """⟨n_{i↑}n_{i↓}⟩ per site, averaged."""
    N = length(psi)
    total = 0.0
    for i in 1:N
        os = OpSum()
        os += 1.0, "Nupdn", i
        docc_op = MPO(os, sites)
        total += real(inner(psi', docc_op, psi))
    end
    return total / N
end


function extract_staggered_mag_sq(psi::MPS, sites::Vector{<:Index})
    """⟨m_s²⟩ = 1/L² Σ_{ij} (-1)^{i+j} ⟨Sᶻ_i Sᶻ_j⟩.

    Uses correlation_matrix("Sz","Sz") — at half-filling with no Zeeman field,
    ⟨Sᶻ_i⟩ = 0 so connected = full correlation.  Correctly includes both
    diagonal (−2⟨docc⟩) and off-diagonal spin−spin terms that the old code
    (product of expectations) missed.
    """
    N = length(psi)
    C_szsz = correlation_matrix(psi, "Sz", "Sz")    # N×N, ⟨Sᶻ_i Sᶻ_j⟩
    total = 0.0
    for i in 1:N
        for j in 1:N
            total += (-1.0)^(i + j) * C_szsz[i, j]
        end
    end
    return total / (N * N)
end


# ══════════════════════════════════════════════════════════════════════
#  4. Dataset generation
# ══════════════════════════════════════════════════════════════════════

function generate_dataset(; L=20,
                            t1_range=(0.1, 2.0),
                            t2_range=(0.1, 2.0),
                            U_range=(0.0, 4.0),
                            V=0.0,
                            n_t1=20, n_t2=20, n_U=5,
                            n_eig_save=20,
                            maxdim=[20, 60, 100, 200, 400, 600, 800, 1000],
                            nsweeps=8, cutoff=1e-8,
                            save_path=nothing,
                            verbose=true)

    t1_arr = collect(range(t1_range[1], stop=t1_range[2], length=n_t1))
    t2_arr = collect(range(t2_range[1], stop=t2_range[2], length=n_t2))
    U_arr  = collect(range(U_range[1], stop=U_range[2], length=n_U))

    n_total = n_t1 * n_t2 * n_U
    if verbose
        println("="^60)
        println("SSH-Hubbard DMRG Dataset Generator")
        println("="^60)
        println("Grid: $(n_t1)x$(n_t2)x$(n_U) = $(n_total) points")
        println("  t1 in [$(t1_range[1]), $(t1_range[2])],  n = $(n_t1)")
        println("  t2 in [$(t2_range[1]), $(t2_range[2])],  n = $(n_t2)")
        println("  U  in [$(U_range[1]), $(U_range[2])],   n = $(n_U)")
        println("  V  = $(V)")
        println("  L  = $(L)   (half-filling, N_up=N_down=$(L÷2))")
        println("  DMRG: maxdim=$(maxdim[end]), nsweeps=$(nsweeps), cutoff=$(cutoff)")
        println()
    end

    # Pre-allocate
    energies = zeros(n_total)
    ent_spectra = zeros(n_total, n_eig_save)
    ent_entropies = zeros(n_total)
    bond_dims = zeros(Int, n_total)
    staggered_charge = zeros(n_total)
    bond_alternation = zeros(n_total)
    double_occup = zeros(n_total)
    stag_mag_sq = zeros(n_total)

    # Pre-build sites (same for all points — ITensor reuses them)
    sites = siteinds("Electron", L; conserve_qns=true)

    # Loop
    t_start = now()
    idx = 0
    for (it1, t1) in enumerate(t1_arr)
        for (it2, t2) in enumerate(t2_arr)
            for (iU, U) in enumerate(U_arr)
                idx += 1

                # --- Ground state (half-filling) ---
                e0, psi = run_dmrg_for_sector(
                    sites, t1, t2, U; V=V,
                    maxdim=maxdim, nsweeps=nsweeps, cutoff=cutoff,
                    outputlevel=0, seed=it1 * 100 + it2 * 10 + iU,
                )
                energies[idx] = e0

                # Bond dimension at the middle bond
                b = L ÷ 2
                orthogonalize!(psi, b + 1)
                bond_dims[idx] = dim(linkind(psi, b))

                # Entanglement observables
                ent_spec, SvN = extract_entanglement(psi, n_eig_save)
                ent_spectra[idx, :] = ent_spec
                ent_entropies[idx] = SvN

                # δn and δB observables
                staggered_charge[idx] = extract_staggered_charge(psi, sites)
                bond_alternation[idx] = extract_bond_alternation(psi, sites)

                # ⟨docc⟩ and ⟨m_s²⟩
                double_occup[idx] = extract_double_occupancy(psi, sites)
                stag_mag_sq[idx] = extract_staggered_mag_sq(psi, sites)

                # Progress
                if verbose && (idx % 10 == 0 || idx == n_total)
                    elapsed = Dates.value(now() - t_start) / 1000
                    pct = idx / n_total * 100
                    eta = elapsed / idx * (n_total - idx)
                    println(
                        "  [$(lpad(idx, 5))/$(n_total)]  " *
                        "$(round(pct, digits=1))%  " *
                        "elapsed $(round(elapsed, digits=0))s  " *
                        "ETA $(round(eta, digits=0))s  " *
                        "E0=$(round(e0, digits=6))  " *
                        "SvN=$(round(SvN, digits=4))  " *
                        "χ=$(bond_dims[idx])  " *
                        "δn=$(round(staggered_charge[idx], digits=2))  " *
                        "δB=$(round(bond_alternation[idx], digits=3))  " *
                        "docc=$(round(double_occup[idx], digits=4))"
                    )
                end
            end
        end
    end

    # --- Reshape to grids ---
    shape_3d = (n_t1, n_t2, n_U)

    data = Dict(
        "t1_arr" => t1_arr,
        "t2_arr" => t2_arr,
        "U_arr" => U_arr,
        "energies" => reshape(energies, shape_3d),
        "ent_spectra" => reshape(ent_spectra, (n_t1, n_t2, n_U, n_eig_save)),
        "ent_entropies" => reshape(ent_entropies, shape_3d),
        "bond_dims" => reshape(bond_dims, shape_3d),
        "staggered_charge" => reshape(staggered_charge, shape_3d),
        "bond_alternation" => reshape(bond_alternation, shape_3d),
        "double_occupancy" => reshape(double_occup, shape_3d),
        "stag_mag_sq" => reshape(stag_mag_sq, shape_3d),
        "L" => L,
        "V" => V,
        "n_eig_save" => n_eig_save,
        "t1_range" => collect(t1_range),
        "t2_range" => collect(t2_range),
        "U_range" => collect(U_range),
        "maxdim" => maxdim,
        "nsweeps" => nsweeps,
        "cutoff" => cutoff,
    )

    # Save
    if save_path !== nothing
        jldsave(save_path; data)
        if verbose
            println("\nSaved -> $(save_path)")
        end
    end

    elapsed_total = Dates.value(now() - t_start) / 1000
    if verbose
        println("  Total time: $(round(elapsed_total, digits=1))s  " *
                "($(round(elapsed_total / n_total, digits=2))s/point)")
    end

    return data
end


# ══════════════════════════════════════════════════════════════════════
#  5. CLI entry point
# ══════════════════════════════════════════════════════════════════════

function main()
    # Defaults
    L = 20
    n_t1, n_t2, n_U = 20, 20, 5
    V = 0.0

    # Parse args
    args = ARGS
    for (i, arg) in enumerate(args)
        if arg == "--L"
            L = parse(Int, args[i + 1])
        elseif arg == "--grid"
            n_t1 = parse(Int, args[i + 1])
            n_t2 = parse(Int, args[i + 2])
            n_U  = parse(Int, args[i + 3])
        elseif arg == "--V"
            V = parse(Float64, args[i + 1])
        elseif arg == "--help"
            println("Usage: julia ssh_dmrg.jl [options]")
            println("  --L <int>       System size (default: 20)")
            println("  --grid n1 n2 n3 Grid points (default: 20 20 5)")
            println("  --V <float>     Nearest-neighbour Coulomb V (default: 0.0)")
            println("  --help          Show this message")
            return
        end
    end

    this_dir = @__DIR__
    save_path = joinpath(this_dir, "dmrg_dataset_L$(L)_$(n_t1)x$(n_t2)x$(n_U).jld2")

    generate_dataset(;
        L=L,
        n_t1=n_t1, n_t2=n_t2, n_U=n_U,
        V=V,
        maxdim=[20, 60, 100, 200, 400, 600, 800, 1000],
        nsweeps=8,
        cutoff=1e-8,
        save_path=save_path,
        verbose=true,
    )
end

if !isdefined(Base, :active_repl)
    main()
end
