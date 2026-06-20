# tdvp_init.jl
#
# TDVP initialization: TDVPInfo diagnostics struct, environment setup,
# composition-order constants and resolution.

# ── Diagnostics ───────────────────────────────────────────────────────────────

"""
    TDVPInfo

Mutable diagnostics container populated by `tdvp_step!`.

Fields:
- `bond_dims_before/after` — MPS bond dimensions before/after the step
- `elapsed`                — total wall time
- `forward_sweep_elapsed`  — time in forward sweeps
- `reverse_sweep_elapsed`  — time in reverse sweeps
- `site_update_elapsed`    — time in 1-site local updates
- `bond_backward_elapsed`  — time in 0-site backward steps
- `gauge_qr_elapsed`       — time in QR/LQ gauge moves
- `env_advance_elapsed`    — time in environment build/advance
- `site_numops`            — Krylov matvec counts for 1-site updates
- `bond_numops`            — Krylov matvec counts for 0-site backward updates
- `site_order`             — `1` for 1-site TDVP, `2` for 2-site TDVP
"""
mutable struct TDVPInfo
    bond_dims_before      :: Vector{Int}
    bond_dims_after       :: Vector{Int}
    elapsed               :: Float64
    forward_sweep_elapsed :: Float64
    reverse_sweep_elapsed :: Float64
    site_update_elapsed   :: Float64
    bond_backward_elapsed :: Float64
    gauge_qr_elapsed      :: Float64
    env_advance_elapsed   :: Float64
    site_numops           :: Vector{Int}
    bond_numops           :: Vector{Int}
    site_order            :: Int
end

TDVPInfo(; site_order::Int = 1) = TDVPInfo(
    Int[], Int[], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Int[], Int[], site_order,
)

function _tdvp_schedule_for_step(step_mode_cfg, step_index::Int)
    if isodd(step_index)
        if step_mode_cfg.step_mode === :third_order_frf
            return TDVP_THIRD_ORDER_RFR
        elseif step_mode_cfg.step_mode === :third_order_rfr
            return TDVP_THIRD_ORDER_FRF
        end
    end
    return step_mode_cfg.sweep_schedule
end

# ── Composition-order constants ───────────────────────────────────────────────

const TDVP_SYMMETRIC_FR = ((:forward, 0.5), (:reverse, 0.5))
const TDVP_SYMMETRIC_RF = ((:reverse, 0.5), (:forward, 0.5))

# Palindromic 2nd-order FRF (coefficients sum to 1; a=c for time-reversibility).
# Labelled "third_order" for API compatibility but genuinely 2nd-order.
const TDVP_THIRD_ORDER_FRF = (
    (:forward, 0.25),
    (:reverse, 0.5),
    (:forward, 0.25),
)
const TDVP_THIRD_ORDER_RFR = (
    (:reverse, 0.25),
    (:forward, 0.5),
    (:reverse, 0.25),
)

const _TDVP_Y1 = 1.0 / (2.0 - cbrt(2.0))
const _TDVP_Y0 = -cbrt(2.0) * _TDVP_Y1
const TDVP_FOURTH_ORDER_YOSHIDA_FR = (
    (:forward, _TDVP_Y1 / 2),
    (:reverse, _TDVP_Y1 / 2),
    (:forward, _TDVP_Y0 / 2),
    (:reverse, _TDVP_Y0 / 2),
    (:forward, _TDVP_Y1 / 2),
    (:reverse, _TDVP_Y1 / 2),
)
const TDVP_FOURTH_ORDER_YOSHIDA_RF = (
    (:reverse, _TDVP_Y1 / 2),
    (:forward, _TDVP_Y1 / 2),
    (:reverse, _TDVP_Y0 / 2),
    (:forward, _TDVP_Y0 / 2),
    (:reverse, _TDVP_Y1 / 2),
    (:forward, _TDVP_Y1 / 2),
)

"""
    _resolve_tdvp_step_mode(step_mode) -> NamedTuple

Resolve a public TDVP step mode symbol to its sweep schedule.

Supported modes:
- `:symmetric_fr`, `:symmetric_rf`                — 2nd order
- `:third_order_frf`, `:third_order_rfr`          — 3rd order
- `:fourth_order_yoshida_fr`, `:fourth_order_yoshida_rf` — 4th order
"""
function _resolve_tdvp_step_mode(step_mode::Symbol)
    if step_mode === :symmetric_fr
        schedule = TDVP_SYMMETRIC_FR
    elseif step_mode === :symmetric_rf
        schedule = TDVP_SYMMETRIC_RF
    elseif step_mode === :third_order_frf
        schedule = TDVP_THIRD_ORDER_FRF
    elseif step_mode === :third_order_rfr
        schedule = TDVP_THIRD_ORDER_RFR
    elseif step_mode === :fourth_order_yoshida_fr
        schedule = TDVP_FOURTH_ORDER_YOSHIDA_FR
    elseif step_mode === :fourth_order_yoshida_rf
        schedule = TDVP_FOURTH_ORDER_YOSHIDA_RF
    else
        error("Unknown TDVP step_mode: $step_mode. " *
              "Supported: :symmetric_fr, :symmetric_rf, :third_order_frf, " *
              ":third_order_rfr, :fourth_order_yoshida_fr, :fourth_order_yoshida_rf.")
    end
    return (step_mode = step_mode, sweep_schedule = schedule)
end
