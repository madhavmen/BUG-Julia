"""Export <Sz_j(t)> from the verified Alice discarded-projector BUG for Julia parity.

The Julia test tests/crosscheck/test_python_parity.jl consumes the JSON this writes.
Both sides must use the SAME model, IC, dt, order, cap and normalize setting -- a
mismatch shows up as a small gap that looks algorithmic but is not, so every one of
those settings is written into the JSON and re-asserted on the Julia side.

Alice is pinned at the SHA recorded in the plan (Task 2); the SHA is captured here so
a regenerated reference can never be silently compared against a different Alice.
"""
import json
import subprocess
import sys

sys.path.insert(0, "/home/madhav.menon/Alice/src")
sys.path.insert(0, "/home/madhav.menon/alice_imagtime_study/xx_sim")

import numpy as np
from alice.algorithm import two_site_bug
from alice.algorithm.two_site_bug.bond import to_complex
from xx_common import (build_bosonic_xx, xx_interactions,
                       domain_wall_config, domain_wall_mps,
                       sz_bit_columns, sz_profile_from_vec)
from imag_time_runner import mps_to_vector

L, DT, NSTEPS, DELTA, MAXDIM = 6, 0.01, 5, 1.0, 8   # delta=1 => Heisenberg


def sz_profile(state, charges, bitcols):
    """<Sz_j> for every site, via this codebase's own dense-vector path.

    The plan's version used `Tensor.to_dense()`, which nicole 0.4.0 does not have.
    `mps_to_vector` + `sz_profile_from_vec` is the route xx_common itself uses, and
    it is already correct about the two conventions that matter for parity:
    site 0 is the MOST significant bit, and local index 0 is up (Sz = +1/2).
    Both match the Julia side (`dense_reference.jl::dense_state`), so the two
    profiles are directly comparable site by site with no relabelling.

    Densifying to 2^L is fine here -- this is a reference export, not the
    integrator, and L = 6 means a 64-element vector.
    """
    vec = mps_to_vector(state, charges).numpy()
    return [float(x) for x in sz_profile_from_vec(vec, bitcols)]


def main():
    interactions, spc = xx_interactions(L, "U1", DELTA)
    ops = build_bosonic_xx("U1", 0.5, DELTA)[1]
    mps = domain_wall_mps(L, spc, ops, domain_wall_config(L))
    for s in range(mps.L):
        mps[s] = to_complex(mps[s])

    charges = [sec.charge for sec in spc.sectors]
    bitcols = sz_bit_columns(L)
    sz0 = sz_profile(mps, charges, bitcols)

    result = two_site_bug.run(mps, interactions, two_site_bug.Options(
        variant="discarded", order="strang", dt=DT, n_steps=NSTEPS,
        max_bond=MAXDIM, trunc_thresh=1e-14, normalize=False))
    state = getattr(result, "state", mps)

    sz = sz_profile(state, charges, bitcols)

    sha = subprocess.check_output(
        ["git", "-C", "/home/madhav.menon/Alice", "rev-parse", "HEAD"]).decode().strip()

    out = {
        "alice_sha": sha,
        "L": L, "dt": DT, "n_steps": NSTEPS, "delta": DELTA, "maxdim": MAXDIM,
        "order": "strang", "trunc_thresh": 1e-14, "normalize": False,
        "variant": "discarded",
        "sz_initial": sz0,
        "sz_final": sz,
    }
    path = "/home/madhav.menon/BUG-Julia/tests/crosscheck/reference_l6_heisenberg.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print("alice sha :", sha)
    print("sz initial:", sz0)
    print("sz final  :", sz)
    print("sum sz    :", sum(sz))
    print("wrote", path)


if __name__ == "__main__":
    main()
