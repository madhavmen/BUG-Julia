# DO THE EXAMPLE SCRIPTS RUN, IN EVERY SYMMETRY MODE, FOR SOMEONE WHO JUST INSTALLED?
#
# Run:  julia --project=. scripts/smoke_examples.jl
# Exit: 0 if every combination ran and produced a well-formed CSV, 1 otherwise.

using Printf

const HERE    = dirname(dirname(abspath(@__FILE__)))
const RESULTS = joinpath(HERE, "examples", "results")

# (script, symmetry, extra args). Small L, short time, tight cap -- seconds, not minutes.
const SMALL = ["L=8", "dt=0.05", "maxdim=16", "sample_every=0.1"]
const CASES = [("xx_domain_wall.jl", "U1",   [SMALL...; "t_max=0.3"]),
               ("xx_domain_wall.jl", "none", [SMALL...; "t_max=0.3"]),
               ("tfim_z2.jl",        "Z2",   [SMALL...; "t_max=0.3"]),
               ("tfim_z2.jl",        "none", [SMALL...; "t_max=0.3"]),
               # OAT sizes its own cap from the exact ceiling, so `maxdim=0` is its default and
               # `t_over_pi` replaces `t_max`.
               ("oat_z2.jl",         "Z2",   ["L=8", "dt=0.05", "t_over_pi=0.2",
                                              "sample_every=0.1"]),
               ("oat_z2.jl",         "none", ["L=8", "dt=0.05", "t_over_pi=0.2",
                                              "sample_every=0.1"])]

"Check the CSV a run produced: header present, rows present, every `err` finite and not absurd."
function check_csv(path)
    isfile(path) || return (false, "no CSV at $(basename(path))")
    lines = readlines(path)
    length(lines) >= 2 || return (false, "CSV has no data rows")
    hdr = split(lines[1], ',')
    ("err" in hdr && "scheme" in hdr) || return (false, "CSV header missing err/scheme")
    ie, isch = findfirst(==("err"), hdr), findfirst(==("scheme"), hdr)
    errs, schemes = Float64[], String[]
    for ln in lines[2:end]
        f = split(ln, ',')
        length(f) == length(hdr) || return (false, "ragged CSV row")
        e = tryparse(Float64, f[ie])
        e === nothing && return (false, "unparseable err $(repr(f[ie]))")
        push!(errs, e); push!(schemes, f[isch])
    end
    all(isfinite, errs) || return (false, "non-finite err (NaN/Inf) in CSV")
    # ⛔ A SILENTLY WRONG RUN STILL EXITS 0 AND STILL WRITES ITS CSV -- measured at 1.9699e-01 on
    # TFIM with the width restoration off. So "the file exists" is not the check; the numbers have
    # to be small enough to mean the integrator ran rather than diverged.
    maximum(errs) < 0.5 || return (false, @sprintf("err too large (%.3e) -- ran but diverged",
                                                   maximum(errs)))
    length(unique(schemes)) == 3 ||
        return (false, "expected 3 schemes, got $(length(unique(schemes)))")
    return (true, @sprintf("%d rows, %d schemes, max err %.2e",
                           length(errs), length(unique(schemes)), maximum(errs)))
end

before = Set(isdir(RESULTS) ? readdir(RESULTS) : String[])
fails  = String[]
@printf("%-22s %-6s %-8s %s\n", "script", "sym", "status", "detail")
println("-"^78)

for (script, sym, extra) in CASES
    args = [extra...; "symmetry=$sym"]
    cmd  = `julia --project=$HERE $(joinpath(HERE, "examples", script)) $args`
    out  = IOBuffer()
    ok   = try
        run(pipeline(cmd; stdout = out, stderr = out)); true
    catch
        false
    end
    if !ok
        s = String(take!(out))
        @printf("%-22s %-6s %-8s %s\n", script, sym, "EXIT!=0",
                replace(last(split(strip(s), '\n'), 1)[1], r"\s+" => " ")[1:min(end, 90)])
        push!(fails, "$script/$sym: nonzero exit")
        continue
    end
    # the CSV this run just wrote is whichever file is new in results/
    now  = Set(readdir(RESULTS))
    fresh = sort(collect(setdiff(now, before)))
    union!(before, now)
    csv = findlast(f -> endswith(f, ".csv"), fresh)
    if csv === nothing
        @printf("%-22s %-6s %-8s %s\n", script, sym, "NO CSV", "run exited 0 but wrote nothing")
        push!(fails, "$script/$sym: no CSV")
        continue
    end
    good, detail = check_csv(joinpath(RESULTS, fresh[csv]))
    @printf("%-22s %-6s %-8s %s\n", script, sym, good ? "ok" : "BAD CSV", detail)
    good || push!(fails, "$script/$sym: $detail")
end

println("-"^78)
if isempty(fails)
    println("ALL $(length(CASES)) COMBINATIONS RAN AND PRODUCED WELL-FORMED CSVs")
    exit(0)
else
    println("$(length(fails)) FAILURE(S):"); foreach(f -> println("  - $f"), fails)
    exit(1)
end
