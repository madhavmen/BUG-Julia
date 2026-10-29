# benchmarks/xx_efficiency/common.jl -- what the Fig. 1-3 drivers of
# xx_domain_wall_cbe_bug_efficiency_standalone_plan.md share: the fixtures, the serialised Fixture M
# snapshots (snapshots.jl) and one raw CSV row format.
#
# Env: XXE_SNAP (the snapshots.jl output dir), XXE_OUT (.), XXE_L (16)

using LinearAlgebra, Printf, JSON, Serialization, Random
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
const RC = BUGJulia.RSVDCBEBondUpdate

const COMMON = joinpath(@__DIR__, "..", "..", "tests", "common")
for f in ("free_fermion.jl", "spin_ed.jl", "ramped_tdvp.jl", "adaptive_fixtures.jl", "dd_fixtures.jl",
          "xx_efficiency_fixtures.jl")
    include(joinpath(COMMON, f))
end

const L = parse(Int, get(ENV, "XXE_L", "16"))
const SNAP = get(ENV, "XXE_SNAP", ".")
const OUT = get(ENV, "XXE_OUT", ".")
const SEEDS = (11, 22, 33)                 # plan Sec. 3.4
const TOLS = (1e-3, 1e-5, 1e-7)            # plan Sec. 3.5

set_symmetry!(:U1)
mkpath(OUT)
const MPO_XX = xxz_mpo(L; J = 1.0, delta = 0.0)

load_snapshot(t, r) = deserialize(joinpath(SNAP, "snap_L$(L)_t$(t)_r$(r).jls"))

"`XXE_CASES` as `t:rank:bond` triples (plan Sec. 2: predetermined snapshots and bonds)."
function cases(default)
    map(split(get(ENV, "XXE_CASES", default), ",")) do s
        a = split(s, ":")
        (t = parse(Float64, a[1]), rank = parse(Int, a[2]), bond = parse(Int, a[3]))
    end
end

"The step's own allocation, as `cbe_bug_step!` builds it (`rule = :linear`, unit norm)."
step_alloc(tol, h) = basis_allocation(AdaptiveBudget(tol, 1e-10, 1e-12), L - 1;
                                      tau = ComplexF64(-h * im), nrm = 1.0, rule = :linear)

"Wall time and bytes of `f()`, warmed once (plan Sec. 3.4: medians of three warmed repetitions)."
function timed(f; reps = 3)
    f()
    ts, bs, out = Float64[], Int[], nothing
    for _ in 1:reps
        GC.gc(false)
        b = @allocated (t = @elapsed (out = f()))
        push!(ts, t); push!(bs, b)
    end
    return out, sort(ts)[(reps + 1) ÷ 2], sort(bs)[(reps + 1) ÷ 2]
end

const CSV_COLS = ("figure", "case", "side", "arm", "param", "tol", "h", "seed", "work", "work_kind",
                  "columns", "error", "error_kind", "target", "pass", "status", "time_s", "bytes")

# Fig 3's `param` ("step=1,round=2") holds a comma: unquoted, it shifted every later column.
_csv_field(s) = occursin(r"[,\"\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s

"Append rows (Dicts keyed by `CSV_COLS`) to `OUT/raw_component_costs.csv`."
function write_rows(rows; file = joinpath(OUT, "raw_component_costs.csv"))
    new = !isfile(file)
    open(file, "a") do io
        new && println(io, join(CSV_COLS, ","))
        for r in rows
            println(io, join((_csv_field(string(get(r, k, ""))) for k in CSV_COLS), ","))
        end
    end
end
