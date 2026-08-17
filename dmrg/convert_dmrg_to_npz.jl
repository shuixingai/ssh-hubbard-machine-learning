import Pkg; Pkg.add("NPZ")
import JLD2
import NPZ

this_dir = @__DIR__   # 脚本自身目录（jld2 同目录），跨机器可移植
files = filter(f -> endswith(f, ".jld2"), readdir(this_dir))
sort!(files; by=f -> begin m = match(r"_(\d+)x(\d+)x(\d+)", f); m === nothing ? 0 : parse(Int, m[1]) * parse(Int, m[2]) end)

fpath = joinpath(this_dir, files[end])
println("Converting: $fpath")

f = JLD2.jldopen(fpath, "r")
d = f["data"]

out = Dict{String, Any}()
for k in keys(d)
    v = d[k]
    if v isa AbstractArray || v isa Number
        out[k] = v
    end
end
JLD2.close(f)

outpath = joinpath(this_dir, "dmrg_dataset.npz")
NPZ.npzwrite(outpath, out)
println("Saved -> $outpath")
println("Keys: ", collect(keys(out)))
