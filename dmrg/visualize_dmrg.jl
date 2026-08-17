"""
DMRG Visualization — PCA on entanglement spectrum + phase diagrams
==================================================================
Reads JLD2 dataset and produces:
  1. PCA/t-SNE on entanglement spectrum (replaces corr-matrix PCA from ED)
  2. Phase diagrams (gap4 label, charge gap label)
  3. Entanglement entropy S_vN landscape
  4. Summary: topological fraction, gap4, S_vN vs U

Usage:
  julia visualize_dmrg.jl
  julia visualize_dmrg.jl --file path/to/data.jld2
"""

using JLD2
using Plots
using MultivariateStats
using ManifoldLearning
using Statistics

# ── Parse args ─────────────────────────────────────────────────────────
function find_dataset()
    this_dir = @__DIR__
    files = filter(f -> endswith(f, ".jld2"), readdir(this_dir))
    isempty(files) && error("No .jld2 files in $this_dir")
    sort!(files; by=f -> begin
        m = match(r"_(\d+)x(\d+)x(\d+)", f)
        m === nothing ? 0 : parse(Int, m[1]) * parse(Int, m[2])
    end)
    return joinpath(this_dir, files[end])
end

data_path = find_dataset()
for (i, arg) in enumerate(ARGS)
    if arg == "--file"; data_path = ARGS[i+1]; end
end

println("Loading: $data_path")
f = jldopen(data_path, "r"); d = f["data"]
t1_arr, t2_arr, U_arr = Vector(d["t1_arr"]), Vector(d["t2_arr"]), Vector(d["U_arr"])
L, energies = d["L"], d["energies"]
ent_spec = d["ent_spectra"]   # (N1, N2, N3, 20)
entropy  = d["ent_entropies"] # (N1, N2, N3)
bdims    = d["bond_dims"]
has_gap = "charge_gap" in keys(d)
has_gap && (charge_gap = d["charge_gap"])
close(f)

N1, N2, N3 = size(energies)
n_total = N1 * N2 * N3
println("Grid: $(N1)x$(N2)x$(N3) = $n_total, L=$L")

# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════

# gap4 (Julia 1-based: index 5 = 5th entanglement level)
gap_4 = ent_spec[:, :, :, 5] .- ent_spec[:, :, :, 4]

# Labels
th_g4 = median(gap_4)
label_g4 = gap_4 .< th_g4

if has_gap
    th_cg = 0.5 * maximum(charge_gap[:, :, 1])
    label_cg = charge_gap .< th_cg
end

quality = bdims .>= 10
println("bond_dim < 10: $(count(bdims .< 10)) / $n_total")
println("gap4 label fraction topo: $(round(100*mean(label_g4), digits=1))%")

# ══════════════════════════════════════════════════════════════════════
#  PCA on entanglement spectrum (20-dim -> 2-dim)
# ══════════════════════════════════════════════════════════════════════

# Reshape ent_spec to (n_total, 20)
X = permutedims(reshape(ent_spec, n_total, 20), [2, 1])  # (20, n_total)

# PCA
M = fit(PCA, X; maxoutdim=2)
X_pca = transform(M, X)'  # (n_total, 2)
var_exp = principalvars(M) / sum(principalvars(M))
println("PCA on ent-spec: 20 -> 2 dim (explains $(round(100*sum(var_exp), digits=1))% variance)")

# t-SNE (using ManifoldLearning)
X_tsne = tsne(X'; ndims=2, perplexity=30)

# ══════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════

this_dir = @__DIR__
colors = [:red, :blue]
markers = [:circle, :dtriangle]

# --- Fig 1: PCA + t-SNE colored by gap4 label ---
p1 = plot(layout=(2, 2), size=(1000, 900), titlefontsize=10,
          legend=:best)

for (row, (Xd, title)) in enumerate([
    (X_pca, "PCA on ent-spec"),
    (X_tsne, "t-SNE on ent-spec"),
])
    for col in 1:2
        ax = subplot(2, 2, (row-1)*2 + col)
        code = col == 1 ? 0 : 1
        name = col == 1 ? "trivial" : "topological"
        mask = vec(label_g4) .== code
        scatter!(ax, Xd[mask, 1], Xd[mask, 2];
                 color=colors[col], marker=markers[col],
                 ms=3, alpha=0.4, label=name)
        xlabel!(ax, "PC1")
        ylabel!(ax, "PC2")
        title!(ax, title)
    end
end
plot!(p1, title="DMRG: PCA/t-SNE on entanglement spectrum (gap4 label)")
savefig(p1, joinpath(this_dir, "dmrg_pca_tsne.png"))
println("Saved -> dmrg_pca_tsne.png")

# --- Fig 1b: PCA colored by continuous gap4 value ---
p1b = plot(layout=(1, 2), size=(1000, 400), titlefontsize=10)
for (idx, (Xd, title)) in enumerate([(X_pca, "PCA"), (X_tsne, "t-SNE")])
    ax = subplot(1, 2, idx)
    scatter!(ax, Xd[:, 1], Xd[:, 2];
             zcolor=vec(gap_4), c=:viridis,
             ms=3, alpha=0.5, label="",
             xlabel="Comp 1", ylabel="Comp 2",
             title="$title (color = gap4)")
end
plot!(p1b)
savefig(p1b, joinpath(this_dir, "dmrg_pca_gap4_color.png"))
println("Saved -> dmrg_pca_gap4_color.png")

# --- Fig 2: Phase diagrams ---
ncols = min(N3, 5)
U_plot = round.(Int, range(1, N3, length=ncols))

p2 = plot(layout=(2, ncols), size=(300*ncols, 500), titlefontsize=10)

for (row, (lbl, title)) in enumerate([
    (label_g4, "gap4 label"),
    has_gap ? (label_cg, "charge gap label") : (label_g4, ""),
])
    row == 2 && !has_gap && continue
    for (col, ui) in enumerate(U_plot)
        ax = subplot(2, ncols, (row-1)*ncols + col)
        lbl_plot = Float64.(lbl[:, :, ui])
        bad = .!quality[:, :, ui]
        lbl_plot[bad] .= NaN

        heatmap!(ax, t1_arr, t2_arr, lbl_plot;
                 xlabel="t1", ylabel="t2",
                 title="U=$(U_arr[ui])",
                 clim=(0, 1), c=:RdBu, aspect_ratio=:equal,
                 size=(300, 300))
        # Mark unreliable points
        if any(bad)
            t1g = [t1_arr[i] for i=1:N1, j=1:N2]
            t2g = [t2_arr[j] for i=1:N1, j=1:N2]
            scatter!(ax, t1g[bad], t2g[bad];
                     marker=:x, color=:gray, ms=3, label="")
        end
    end
end
plot!(p2, title="DMRG Phase Diagrams (L=$L)")
savefig(p2, joinpath(this_dir, "dmrg_phase_diagrams.png"))
println("Saved -> dmrg_phase_diagrams.png")

# --- Fig 3: Entropy landscape ---
p3 = plot(layout=(2, 3), size=(1200, 800), titlefontsize=10)
for (idx, ui) in enumerate([1, max(1, N3÷3), max(1, 2*N3÷3), N3])
    idx > 4 && break
    ax = subplot(2, 3, idx)
    heatmap!(ax, t1_arr, t2_arr, entropy[:, :, ui];
             xlabel="t1", ylabel="t2",
             title="SvN  U=$(U_arr[ui])",
             c=:viridis, aspect_ratio=:equal)
end
ax = subplot(2, 3, 5)
t1g = [t1_arr[i] for i=1:N1, j=1:N2]
t2g = [t2_arr[j] for i=1:N1, j=1:N2]
for (ci, ui) in enumerate([1, N3÷2+1, N3])
    scatter!(ax, vec(t2g .- t1g), vec(entropy[:, :, ui]);
             ms=2, alpha=0.3, label="U=$(U_arr[ui])")
end
xlabel!(ax, "t2 - t1"); ylabel!(ax, "SvN")
title!(ax, "Entanglement Entropy")
vline!(ax, [0], color=:gray, ls=:dash, label="")
plot!(p3)
savefig(p3, joinpath(this_dir, "dmrg_entropy.png"))
println("Saved -> dmrg_entropy.png")

# --- Fig 4: Charge gap landscape ---
if has_gap
    p4 = plot(layout=(2, 3), size=(1200, 800), titlefontsize=10)
    for (idx, ui) in enumerate([1, max(1, N3÷3), max(1, 2*N3÷3), N3])
        idx > 4 && break
        ax = subplot(2, 3, idx)
        heatmap!(ax, t1_arr, t2_arr, charge_gap[:, :, ui];
                 xlabel="t1", ylabel="t2",
                 title="Charge gap  U=$(U_arr[ui])",
                 c=:plasma, aspect_ratio=:equal)
    end
    ax = subplot(2, 3, 5)
    for (ci, ui) in enumerate([1, N3÷2+1, N3])
        scatter!(ax, vec(t2g .- t1g), vec(charge_gap[:, :, ui]);
                 ms=2, alpha=0.3, label="U=$(U_arr[ui])")
    end
    xlabel!(ax, "t2 - t1"); ylabel!(ax, "Charge gap")
    title!(ax, "Charge Gap"); vline!(ax, [0], color=:gray, ls=:dash, label="")
    plot!(p4)
    savefig(p4, joinpath(this_dir, "dmrg_charge_gap.png"))
    println("Saved -> dmrg_charge_gap.png")
end

# --- Fig 5: Summary panel ---
p5 = plot(layout=(2, 2), size=(1000, 800), titlefontsize=10)

ax = subplot(2, 2, 1)
frac_g4 = [mean(label_g4[:, :, ui]) for ui in 1:N3]
plot!(ax, U_arr, frac_g4; marker=:o, lw=2, label="gap4", color=:blue)
has_gap && (frac_cg = [mean(label_cg[:, :, ui]) for ui in 1:N3];
    plot!(ax, U_arr, frac_cg; marker=:s, lw=2, label="charge gap", color=:red))
xlabel!(ax, "U"); ylabel!(ax, "Fraction topo"); title!(ax, "Topological fraction vs U")

ax = subplot(2, 2, 2)
mean_S = [mean(entropy[:, :, ui]) for ui in 1:N3]
plot!(ax, U_arr, mean_S; marker=:o, lw=2, color=:green, label="")
xlabel!(ax, "U"); ylabel!(ax, "Mean SvN"); title!(ax, "Entropy vs U")

ax = subplot(2, 2, 3)
mean_g4 = [mean(gap_4[:, :, ui]) for ui in 1:N3]
plot!(ax, U_arr, mean_g4; marker=:o, lw=2, color=:purple, label="")
xlabel!(ax, "U"); ylabel!(ax, "Mean gap4"); title!(ax, "gap4 vs U")

ax = subplot(2, 2, 4)
if has_gap
    mean_cg = [mean(charge_gap[:, :, ui]) for ui in 1:N3]
    plot!(ax, U_arr, mean_cg; marker=:o, lw=2, color=:orange, label="")
    xlabel!(ax, "U"); ylabel!(ax, "Mean charge gap"); title!(ax, "Charge gap vs U")
end

plot!(p5)
savefig(p5, joinpath(this_dir, "dmrg_summary.png"))
println("Saved -> dmrg_summary.png")

println("\nAll figures -> $this_dir")
