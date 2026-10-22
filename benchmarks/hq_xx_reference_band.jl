# benchmarks/hq_xx_reference_band.jl -- the converged L = 16 reference and its uncertainty band
# (HQ plan T7.2c, T7.4) from several `hq_xx_reference.jl` runs.
#
#     julia benchmarks/hq_xx_reference_band.jl OUT.json FINEST.json OTHER1.json OTHER2.json ...
#
# The reference is FINEST's record. The band at each recorded time is the largest deviation of any
# other setting from it, `max_k max_i |m_i^k(t) - m_i^fin(t)|` (and the same for the centre current).
# A band is only a convergence statement if the settings differ in EACH of D_max, dt and the
# truncation tolerance, and in at least two solvers/orders: the script prints which axes were varied
# and refuses (exit 1) when fewer than two are.

using JSON, Printf

length(ARGS) >= 3 || error("usage: OUT.json FINEST.json OTHER.json [OTHER.json ...]")
out, fin_path, others = ARGS[1], ARGS[2], ARGS[3:end]
fin = JSON.parsefile(fin_path)
runs = [JSON.parsefile(p) for p in others]
cf = fin["config"]
for r in runs
    all(r["config"][k] == cf[k] for k in ("twoS", "L", "J")) ||
        error("$(r["config"]) is a different model from $(cf)")
end
axes = [k for k in ("maxdim", "dt", "trunc", "order") if any(r["config"][k] != cf[k] for r in runs)]
println("HQBAND-AXES varied: ", join(axes, ", "))
length(axes) >= 2 || (println("HQBAND-REFUSED fewer than two convergence axes varied"); exit(1))

recorded(run, t) = findfirst(x -> abs(x["t"] - t) < 1e-9, run["records"])
at(run, t) = run["records"][recorded(run, t)]
# A run stopped by its walltime holds the records it reached: the band covers only the times
# EVERY setting reached, and says where that horizon is.
times = [r["t"] for r in fin["records"] if all(x -> recorded(x, r["t"]) !== nothing, runs)]
println("HQBAND-HORIZON t <= $(last(times)) (FINEST reached $(fin["records"][end]["t"]))")
band_m, band_j = zeros(length(times)), zeros(length(times))
for (k, t) in enumerate(times), r in runs
    x, y = at(r, t), at(fin, t)
    band_m[k] = max(band_m[k], maximum(abs.(Float64.(x["m"]) - Float64.(y["m"]))))
    band_j[k] = max(band_j[k], abs(x["j"] - y["j"]))
end
for (r, p) in zip(runs, others)
    dm = maximum(maximum(abs.(Float64.(at(r, t)["m"]) - Float64.(at(fin, t)["m"]))) for t in times)
    @printf("HQBAND-SETTING %s max|dm|=%.3e\n", basename(p), dm)
end
for k in eachindex(times)
    k % 10 == 1 || k == length(times) || continue
    @printf("HQBAND t=%.2f band_m=%.3e band_j=%.3e\n", times[k], band_m[k], band_j[k])
end
open(out, "w") do io
    JSON.print(io, Dict("config" => cf, "settings" => [r["config"] for r in runs],
                        "varied" => axes, "t" => times,
                        "m" => [at(fin, t)["m"] for t in times],
                        "j" => [at(fin, t)["j"] for t in times],
                        "band_m" => band_m, "band_j" => band_j))
end
println("HQBAND-DONE $out max band_m=$(maximum(band_m)) band_j=$(maximum(band_j))")
