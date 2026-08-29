# The L = 16 open-system real-time-evolution campaign: Hryniuk & Szymanska, Comms Phys (2026) 9:45.
# Runs the four figures CONCURRENTLY and records each job's PID.
#
#     pwsh -File benchmarks/run_dissipative_l16.ps1
#
#   fig3    nearest-neighbour anisotropic XYZ + spin decay   (their Fig. 3)   dissipative_heisenberg.jl
#   fig4    diagonal long-range Ising, chain                 (their Fig. 4ab) dissipative_longrange.jl
#   fig4sq  the same on a 4x4 open patch                     (their Fig. 4cd) dissipative_longrange.jl
#   fig6    NON-diagonal long-range XYZ, chain, rmax = 4     (their Fig. 6cd) dissipative_longrange.jl
#
# Fig. 5 (the steady-state S_zz(q) sweep over J2) is NOT here: it needs a converged NESS at every
# J2, which is several times the cost of all four above. Run it deliberately and at a size you have
# the budget for:
#     LR_L=12 LR_J2S=0,0.5,1,1.5,2 LR_STEADY_TMAX=8 julia -t 2 --project=. `
#         benchmarks/dissipative_longrange.jl fig5
#
# ── WHY CONCURRENTLY, AND WHAT IT COSTS ─────────────────────────────────────────────────────────
# These sweeps are sequential and top out near ~1.5 cores each whatever BLAS is told, so four of
# them fit comfortably on a 12-core box and wall clock becomes max-of-runs instead of sum-of-runs.
#
# ⛔ THE PRICE IS THE `seconds` COLUMN, PAID KNOWINGLY. Arm-vs-arm cost must be read off `krylov`
# (cumulative matvecs), which is independent of load and of hardware. `seconds` stays in the CSV as
# a sanity check, never as a result. If you need publication wall-clock, run ONE job at a time.
#
# ⛔ BLAS = 2, MEASURED. Raising it to 4 made things WORSE (fig3: 31.5 s/step vs 20.4) -- chi=20,
# d=4 tensors are far too small to feed four threads, and the extra threads only thrash.
#
# ⚠ THE BOX MUST NOT SLEEP. An earlier attempt lost nine hours of wall clock to suspend: the four
# jobs accumulated 67 CPU-seconds between 22:37 and 07:29. `keepawake.ps1` holds the system up via
# SetThreadExecutionState for as long as IT runs -- it is not a persistent power-setting change and
# leaves nothing behind when killed.

param(
    [int]    $L       = 16,
    [double] $Dt      = 0.05,
    [int]    $Cap     = 20,
    [double] $Tmax    = 4.0,
    [int]    $Nsave   = 21,
    [int]    $Maxiter = 8,
    [string[]] $Only  = @()          # e.g. -Only fig4,fig6 to run a subset
)

$repo = Split-Path -Parent $PSScriptRoot
$res  = Join-Path $repo "benchmarks\results"
New-Item -ItemType Directory -Force -Path $res | Out-Null
Set-Location $repo

$jl = (Get-Command julia -ErrorAction SilentlyContinue).Source
if (-not $jl) { throw "julia not found on PATH" }

# --- stop any jobs this script started before, by RECORDED PID ---------------------------------
# ⛔ NOT by name, start time, or command line. Julia rewrites its own command line, so
# Win32_Process reports a bare `julia.exe` with no arguments and the launching shell has usually
# exited -- an earlier attempt could not tell its own four jobs apart afterwards, and a
# guess-and-kill would have taken out an unrelated session's run.
$pidfile = Join-Path $res "run_pids.txt"
if (Test-Path $pidfile) {
    $known = Get-Content $pidfile |
             Where-Object { $_ -match '^\S+\s+\d+$' } |
             ForEach-Object { [int](($_ -split '\s+')[1]) }
    foreach ($p in (Get-Process julia, powershell -ErrorAction SilentlyContinue |
                    Where-Object { $known -contains $_.Id })) {
        Write-Output ("  stopping previous PID {0} (CPU {1:N0}s)" -f $p.Id, $p.CPU)
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 4
}

# --- keep the machine awake for the duration ----------------------------------------------------
$keepAwake = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "keepawake.ps1"))
Write-Output ("  keep-awake helper PID {0}" -f $keepAwake.Id)

# --- launch --------------------------------------------------------------------------------------
$env:OPENBLAS_NUM_THREADS = "2"
$env:OMP_NUM_THREADS      = "2"
$env:LR_L = "$L"; $env:LR_DT = "$Dt"; $env:LR_CAP = "$Cap"; $env:LR_TMAX = "$Tmax"
$env:LR_NSAVE = "$Nsave"; $env:LR_MAXITER = "$Maxiter"; $env:LR_CHAIN_RMAX = "Inf"
$env:DH_L = "$L"; $env:DH_DT = "$Dt"; $env:DH_CAP = "$Cap"; $env:DH_TMAX = "$Tmax"
$env:DH_NSAVE = "$Nsave"; $env:DH_MAXITER = "$Maxiter"

"# $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  keepawake $($keepAwake.Id)" |
    Out-File $pidfile -Encoding utf8
"keepawake $($keepAwake.Id)" | Out-File $pidfile -Append -Encoding utf8

$jobs = @(
    @{ tag = "fig4";   script = "dissipative_longrange.jl";  mode = "fig4"     },
    @{ tag = "fig6";   script = "dissipative_longrange.jl";  mode = "fig6"     },
    @{ tag = "fig4sq"; script = "dissipative_longrange.jl";  mode = "fig4sq"   },
    @{ tag = "fig3";   script = "dissipative_heisenberg.jl"; mode = "protocol" }
)
foreach ($j in $jobs) {
    if ($Only.Count -gt 0 -and $Only -notcontains $j.tag) { continue }
    $log = Join-Path $res ("run_" + $j.tag + ".log")
    $p = Start-Process -FilePath $jl `
        -ArgumentList @("-t", "2", "--project=.", ("benchmarks/" + $j.script), $j.mode) `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    "$($j.tag) $($p.Id)" | Out-File $pidfile -Append -Encoding utf8
    Write-Output ("  launched {0,-8} PID {1,-7} -> {2}" -f $j.tag, $p.Id, $log)
}
Write-Output "PIDs recorded in $pidfile"
Write-Output "plot with:  python3 benchmarks/plot_dissipative_longrange.py"
