#!/bin/bash
# benchmarks/dissipative_restart_check.sh -- plan §4.3 and the Task 7 scheduler smoke ("dry-run
# output and restart"): dissipative_run.jl run uninterrupted, and stopped at its DR_STOP_AT-th
# checkpoint then resumed, must write the same JSON, measurements aside (wall, phase timers, peak
# RSS, allocated bytes); a resume under a changed tolerance must be refused (CONTROL).
#   usage (from the repository root, the run's DR_* in the environment):
#     dissipative_restart_check.sh PKG OUTDIR STOP_AT
# Prints RESTART-CHECK PASS or FAIL lines and exits 0 only if every check passed.
set -u
PKG=$1; O=$2; STOP=$3
mkdir -p "$O"
run() { julia -t 1 --startup-file=no --project="$PKG" benchmarks/dissipative_run.jl; }
DR_OUT=$O/full.json DR_CKPT=$O/full.ckpt run > "$O/full.log" 2>&1; r_full=$?
DR_OUT=$O/rs.json DR_CKPT=$O/rs.ckpt DR_CKPT_SEC=0 DR_STOP_AT=$STOP run > "$O/stop.log" 2>&1; r_stop=$?
# read now: the resume below writes rs.json and removes rs.ckpt (dr6x)
stopped=$([ $r_stop = 0 ] && grep -q "STOPPED-AT checkpoint $STOP" "$O/stop.log" && [ -s "$O/rs.ckpt" ] && \
          [ ! -e "$O/rs.json" ]; echo $?)
# CONTROL, before the resume removes the checkpoint: another tol_trunc must not resume it
DR_OUT=$O/bad.json DR_CKPT=$O/rs.ckpt DR_TOLT=1e-6 run > "$O/bad.log" 2>&1; r_bad=$?
DR_OUT=$O/rs.json DR_CKPT=$O/rs.ckpt run > "$O/resume.log" 2>&1; r_res=$?
fail=0
check() { if [ "$2" = 0 ]; then echo "RESTART-CHECK PASS $1"; else echo "RESTART-CHECK FAIL $1"; fail=1; fi; }
check "uninterrupted run exits 0 (rc $r_full)" $([ $r_full = 0 ] && [ -s "$O/full.json" ]; echo $?)
check "stopped run wrote checkpoint $STOP and no JSON (rc $r_stop)" $stopped
check "CONTROL: changed tol_trunc refused (rc $r_bad)" \
      $([ $r_bad != 0 ] && grep -q "inconsistent restart" "$O/bad.log" && [ ! -e "$O/bad.json" ]; echo $?)
check "resumed run exits 0 and removed its checkpoint (rc $r_res)" \
      $([ $r_res = 0 ] && grep -q "resuming from" "$O/resume.log" && [ -s "$O/rs.json" ] && \
        [ ! -e "$O/rs.ckpt" ]; echo $?)
julia --startup-file=no --project="$PKG" -e '
using JSON
a, b = JSON.parsefile(ARGS[1]), JSON.parsefile(ARGS[2])
meas(k) = startswith(k, "t_") || k in ("wall", "peak_rss", "bytes")
bad = String[]
for k in sort(collect(union(keys(a), keys(b))))
    k in ("wall", "peak_rss", "restarts", "steplog") && continue
    get(a, k, nothing) == get(b, k, nothing) || push!(bad, k)
end
for k in sort(collect(keys(a["steplog"])))
    meas(k) || a["steplog"][k] == get(b["steplog"], k, nothing) || push!(bad, "steplog." * k)
end
(a["restarts"], b["restarts"]) == (0, 1) || push!(bad, "restarts $((a["restarts"], b["restarts"]))")
println(isempty(bad) ? "RESTART-CHECK PASS resumed JSON equals the uninterrupted one" :
        "RESTART-CHECK FAIL fields differ: $(join(bad, ", "))")
exit(isempty(bad) ? 0 : 1)' "$O/full.json" "$O/rs.json"; r_cmp=$?
[ $r_cmp = 0 ] || fail=1
exit $fail
