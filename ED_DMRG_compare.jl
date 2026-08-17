"""
ED vs DMRG 交叉验证 — DMRG 部分
=====================================
测试点 (t₁=t₂=0.01):
  case A: U=4, V=0, L=6
  case B: U=4, V=0, L=10
  case C: U=0, V=4, L=6
  case D: U=0, V=4, L=10

提取: E0, δn, δB, ⟨(δn)²⟩, 双占据, ⟨m_s²⟩, gap4

用法:  julia ED_DMRG_compare.jl
"""

using ITensors
using ITensorMPS
using LinearAlgebra
using Random
using Statistics

# ══════════════════════════════════════════════════════════════════
#  Hamiltonian builder
# ══════════════════════════════════════════════════════════════════

function build_ham(sites, t1, t2, U, V)
    N = length(sites)
    os = OpSum()
    for j in 1:(N - 1)
        t = isodd(j) ? t1 : t2
        os += -t, "Cdagup", j, "Cup", j + 1
        os += -t, "Cdagup", j + 1, "Cup", j
        os += -t, "Cdagdn", j, "Cdn", j + 1
        os += -t, "Cdagdn", j + 1, "Cdn", j
    end
    for j in 1:N
        os += U, "Nupdn", j
    end
    if V != 0.0
        for j in 1:(N - 1)
            os += V, "Ntot", j, "Ntot", j + 1
        end
    end
    return os
end

# ══════════════════════════════════════════════════════════════════
#  DMRG solver
# ══════════════════════════════════════════════════════════════════

function run_dmrg(sites, t1, t2, U, V;
                  maxdim=[20, 60, 100, 200, 400, 600, 800],
                  nsweeps=7, cutoff=1e-8, seed=42)
    N = length(sites)
    Random.seed!(seed)
    os = build_ham(sites, t1, t2, U, V)
    H = MPO(os, sites)

    state = fill("Emp", N)
    for j in 1:N
        state[j] = isodd(j) ? "Up" : "Dn"
    end
    psi0 = randomMPS(sites, state, 10)
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    return energy, psi
end

# ══════════════════════════════════════════════════════════════════
#  Observables
# ══════════════════════════════════════════════════════════════════

function delta_n(psi, sites)
    N = length(psi)
    n_all = expect(psi, "Ntot")
    total = 0.0
    for i in 1:N
        total += (-1.0)^i * n_all[i]
    end
    return total / N
end

function delta_n_sq(psi, sites)
    """⟨(δn)²⟩ from MPS — compute ⟨n_i n_j⟩ via local measurements.

    Uses the MPS to compute the full operator expectation exactly,
    accounting for all density-density correlations.
    """
    N = length(psi)
    # Get all site occupations simultaneously
    n_all = expect(psi, "Ntot")

    # Build ⟨n_i n_j⟩: for i=j, n_i² = n_i (since n_i ∈ {0,1,2} but
    # at half-filling ⟨n_i⟩ = 1, we need the actual value)
    # Actually for i=j, n_i² ≠ n_i because n_i can be 0, 1, or 2.
    # n_i² = n_i + 2 n_{i↑}n_{i↓} (since (n↑+n↓)² = n↑² + n↓² + 2n↑n↓ = n↑+n↓+2n↑n↓)
    # We need to measure these separately.

    total = 0.0
    for i in 1:N
        for j in 1:N
            if i == j
                # ⟨n_i²⟩ = ⟨n_i⟩ + 2⟨n_{i↑}n_{i↓}⟩
                n_i_tot = n_all[i]
                # measure double occupancy for site i
                os = OpSum()
                os += 1.0, "Nupdn", i
                docc_op = MPO(os, sites)
                docc_i = real(inner(psi', docc_op, psi))
                n_i_sq = n_i_tot + 2.0 * docc_i
                total += (-1.0)^(i + i) * n_i_sq
            else
                # ⟨n_i n_j⟩ = measure as product via MPO
                os = OpSum()
                os += 1.0, "Ntot", i, "Ntot", j
                nn_op = MPO(os, sites)
                nn_ij = real(inner(psi', nn_op, psi))
                total += (-1.0)^(i + j) * nn_ij
            end
        end
    end
    return total / (N * N)
end

function delta_B(psi, sites)
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

function double_occupancy(psi, sites)
    N = length(psi)
    D = zeros(N)
    for i in 1:N
        os = OpSum()
        os += 1.0, "Nupdn", i
        docc_op = MPO(os, sites)
        D[i] = real(inner(psi', docc_op, psi))
    end
    return D
end

function staggered_mag_sq(psi, sites)
    """⟨m_s²⟩ = 1/L² Σ_{ij} (-1)^{i+j} ⟨Sᶻ_i Sᶻ_j⟩.

    Uses correlation_matrix for correct diagonal (−2⟨docc⟩) and off-diagonal
    (spin−spin correlation) terms.
    """
    N = length(psi)
    C_szsz = correlation_matrix(psi, "Sz", "Sz")
    total = 0.0
    for i in 1:N
        for j in 1:N
            total += (-1.0)^(i + j) * C_szsz[i, j]
        end
    end
    return total / (N * N)
end

function extract_entanglement(psi, n_max=20)
    """Return (ent_spec, SvN)."""
    N = length(psi)
    b = N ÷ 2
    orthogonalize!(psi, b + 1)
    li = linkind(psi, b)
    U_mat, S_diag, V_mat = svd(psi[b + 1], li)
    lambdas = Vector(diag(S_diag))
    lambdas = lambdas[lambdas .> 1e-14]
    lambda_sq = Vector(lambdas .^ 2)
    lambda_sq = lambda_sq ./ sum(lambda_sq)
    SvN = -sum(lambda_sq .* log.(max.(lambda_sq, 1e-30)))
    epsilon = Vector(-log.(max.(lambda_sq, 1e-30)))
    sort!(epsilon)
    n = min(length(epsilon), n_max)
    out = zeros(n_max)
    out[1:n] = epsilon[1:n]
    return out, SvN
end

# ══════════════════════════════════════════════════════════════════
#  Test cases
# ══════════════════════════════════════════════════════════════════

# ── 命令行参数：如果传了参数则自定义，否则默认 4 个标准测试点 ──
if length(ARGS) >= 3
    global t1 = parse(Float64, ARGS[1])
    global t2 = parse(Float64, ARGS[2])
    rest = ARGS[3:end]
    global cases = Tuple{Int,Float64,Float64,String}[]
    for k in 1:4:length(rest)
        L = parse(Int, rest[k])
        U = parse(Float64, rest[k+1])
        V = parse(Float64, rest[k+2])
        if k+3 <= length(rest)
            label = rest[k+3]
        else
            label = "L=$L U=$U V=$V"
        end
        push!(cases, (L, U, V, label))
    end
else
    t1 = 0.01
    t2 = 0.01
    cases = [
        (6,  4.0, 0.0, "Mott L=6"),
        (10, 4.0, 0.0, "Mott L=10"),
        (6,  0.0, 4.0, "CDW L=6"),
        (10, 0.0, 4.0, "CDW L=10"),
    ]
end

println("="^72)
println("DMRG 计算: t₁ = $(t1), t₂ = $(t2)")
println("="^72)

for (L, U, V, label) in cases
    println("\n  --- $(label) (L=$L, U=$U, V=$V) ---")

    sites = siteinds("Electron", L; conserve_qns=true)
    e0, psi = run_dmrg(sites, t1, t2, U, V)

    dn_val = delta_n(psi, sites)
    dn2    = delta_n_sq(psi, sites)
    dB_val = delta_B(psi, sites)
    docc   = double_occupancy(psi, sites)
    ms2    = staggered_mag_sq(psi, sites)
    ent, SvN = extract_entanglement(psi)
    gap4   = (length(ent) >= 5) ? (ent[5] - ent[4]) : NaN

    println("    E0      = $(round(e0, digits=10))")
    println("    δn      = $(round(dn_val, digits=10))")
    println("    ⟨(δn)²⟩ = $(round(dn2, digits=10))")
    println("    (⟨δn⟩)² = $(round(dn_val^2, digits=10))")
    println("    δB      = $(round(dB_val, digits=6))")
    println("    ⟨docc⟩  = $(round(mean(docc), digits=6))")
    println("    ⟨m_s²⟩  = $(round(ms2, digits=10))")
    println("    gap4    = $(round(gap4, digits=6))")
    println("    SvN     = $(round(SvN, digits=6))")
    println("    occup   = [$(join(round.(expect(psi, "Ntot"), digits=4), " "))]")
end

println("\n" * "="^72)
println("Done. 对比 ED 结果: python ED_DMRG_compare.py")
println("="^72)
