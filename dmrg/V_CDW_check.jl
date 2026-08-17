# V-CDW 验证脚本：DMRG 检验 L=6 和 L=20 下 V 是否产生 CDW
# 运行:  julia V_CDW_check.jl
using ITensors
using ITensorMPS
using LinearAlgebra
using Random

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

function run_dmrg(sites, t1, t2, U, V; maxdim=[20, 60, 100, 200, 400, 600])
    N = length(sites)
    Random.seed!(42)
    os = build_ham(sites, t1, t2, U, V)
    H = MPO(os, sites)

    state = fill("Emp", N)
    for j in 1:N
        state[j] = isodd(j) ? "Up" : "Dn"
    end
    psi0 = randomMPS(sites, state, 10)
    energy, psi = dmrg(H, psi0; nsweeps=6, maxdim=maxdim, cutoff=1e-8, outputlevel=0)
    return energy, psi
end

function delta_n(psi, sites)
    N = length(psi)
    n_all = expect(psi, "Ntot")  # vector of site occupations
    total = 0.0
    for i in 1:N
        total += (-1.0)^i * n_all[i]
    end
    return total / N
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

function site_occupations(psi, sites)
    return expect(psi, "Ntot")
end

# ═══════════════════════════════════
#  Parameters
# ═══════════════════════════════════
cases = [
    (6,  4.0, "L=6,  V=4"),
    (6,  8.0, "L=6,  V=8"),
    (20, 4.0, "L=20, V=4"),
    (20, 8.0, "L=20, V=8"),
]

println("="^65)
println("CDW verification: t1=t2=1, U=0, varying V and L")
println("="^65)

for (L, V, label) in cases
    println("\n  --- $label ---")
    sites = siteinds("Electron", L; conserve_qns=true)
    e0, psi = run_dmrg(sites, 1.0, 1.0, 0.0, V)

    dn = delta_n(psi, sites)
    dB = delta_B(psi, sites)
    occ = site_occupations(psi, sites)

    println("  E0  = $e0")
    println("  δn  = $dn")
    println("  δB  = $dB")
    println("  n_i = [$(join(round.(occ, digits=6), " "))]")
    println("  Σ n_i = $(sum(occ))")
end

println("\n" * "="^65)
println("Done.")
