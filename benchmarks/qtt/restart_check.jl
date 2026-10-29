# benchmarks/qtt/restart_check.jl -- plan §4.4: an interrupted-and-resumed production run must
# reproduce the uninterrupted one bitwise (or within a stated deterministic tolerance).
# ENV: RC_A, RC_B (two production.jl output dirs at the same config, one stopped by P_STOP_AT and
# resumed).

using Serialization, LinearAlgebra
using BUGJulia.RSVDCBEBondUpdate: tt_dense, tt_norm, tt_add, tt_scale

a = deserialize(joinpath(ENV["RC_A"], "ckpt.jls"))
b = deserialize(joinpath(ENV["RC_B"], "ckpt.jls"))
a.config_hash == b.config_hash || error("different configs")
println("t: $(a.t) vs $(b.t); steps $(a.k) vs $(b.k)")
bitwise = length(a.cores) == length(b.cores) && all(a.cores[i] == b.cores[i] for i in eachindex(a.cores))
d = tt_norm(tt_add(a.cores, tt_scale(b.cores, -1.0))) / tt_norm(b.cores)
println("RESTART bitwise = $bitwise  relative distance = $d")
la, lb = a.ledger, b.ledger
same_h = length(la) == length(lb) && all(la[i].h == lb[i].h for i in eachindex(la))
println("RESTART step sizes identical = $same_h")
(bitwise || d < 1e-12) && a.t == b.t || error("RESTART CHECK FAILED")
println("RESTART-OK")
