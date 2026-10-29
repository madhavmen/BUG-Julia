# benchmarks/dissipative_b2_sector_p0.py -- exact no-jump survival P0(t) of the Task 7 B2 trajectory start
# (L = 32, 4 excitations, alt, Jz = 1, Gamma = 2, ka = pi/3; dissipative_run.jl build_model). The no-jump
# generator H - iQ/2 conserves the excitation number, so the sector is C(32,4) = 35960 states: sparse, no
# Liouvillian. Graded the dr7w row 9/10 trajectories' first steps (audit, Task 7 B2 trajectories).
import itertools, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
L, N, Jz, G, Gm, k = 32, 4, 1.0, 2.0, 2.0, np.pi/3
x = np.arange(1, L+1)
states = list(itertools.combinations(range(L), N)); idx = {s: i for i, s in enumerate(states)}
h = np.array([0.2*(-1)**j for j in range(1, L+1)])
c = G*np.exp(1j*k*(x[None,:]-x[:,None])) + Gm*np.exp(-1j*k*(x[None,:]-x[:,None]))  # c[i,j]
rows, cols, vals = [], [], []
for a, s in enumerate(states):
    occ = set(s); sz = np.array([0.5 if j in occ else -0.5 for j in range(L)])
    d = Jz*np.sum(sz[:-1]*sz[1:]) + np.sum(h*sz) - 0.5j*sum(c[j,j].real for j in s)
    rows.append(a); cols.append(a); vals.append(d)
    for j in s:                      # move the up spin at j to i: S+_i S-_j
        for i in range(L):
            if i in occ: continue
            t = tuple(sorted((occ - {j}) | {i})); b = idx[t]
            v = -0.5j*c[i, j]
            if abs(i-j) == 1: v += 0.5
            rows.append(b); cols.append(a); vals.append(v)
Heff = sp.csr_matrix((vals, (rows, cols)), shape=(len(states),)*2)
psi = np.zeros(len(states), complex); psi[idx[(0, 2, 4, 6)]] = 1
ts = [0.0125, 0.025, 0.05, 0.1, 0.15, 0.2, 0.5, 1.0]
out = expm_multiply(-1j*Heff, psi, start=0, stop=1.0, num=81, endpoint=True)
grid = np.linspace(0, 1, 81)
for t in ts:
    v = out[int(round(t/ (1/80)))] if abs(t/(1/80) - round(t/(1/80))) < 1e-9 else expm_multiply(-1j*Heff*t, psi)
    print("t=%.4f  P0=%.6f  logP0=%.5f" % (t, np.vdot(v, v).real, np.log(np.vdot(v, v).real)))
print("initial rate <Q> =", sum(c[j,j].real for j in (0,2,4,6)))
