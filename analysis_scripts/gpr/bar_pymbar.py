#!/usr/bin/env python3
import sys, os
import numpy as np
from pymbar.other_estimators import bar

if len(sys.argv) < 4:
    print("Usage: bar_pymbar.py SEED.npz --beta BETA [--ndata N]")
    sys.exit(1)

seed_file = sys.argv[1]
if not os.path.exists(seed_file):
    sys.exit(f"[ERROR] Seed file not found: {seed_file}")

if "--beta" not in sys.argv:
    sys.exit("[ERROR] --beta is required")
beta = float(sys.argv[sys.argv.index("--beta")+1])

ndata = None
if "--ndata" in sys.argv:
    ndata = int(sys.argv[sys.argv.index("--ndata")+1])

print(f"[BAR:pymbar] Loading seed from {seed_file}")
data = np.load(seed_file, allow_pickle=True)

try:
    wF = data["wF"]
    wR = data["wR"]
except KeyError as e:
    sys.exit(f"[ERROR] Missing key: {e}")

if ndata is not None:
    wF = wF[:ndata]
    wR = wR[:ndata]
    print(f"[BAR:pymbar] Using first {ndata} datapoints")

print(f"[INFO] Totals: F={len(wF)}, R={len(wR)}")
if len(wF) == 0 or len(wR) == 0:
    sys.exit("[ERROR] No work data found in seed file")

print(f"[DEBUG] First works: F={wF[:5]} R={wR[:5]}")

# ---- pymbar BAR ----
results = bar(wF, wR, DeltaF=0.0, compute_uncertainty=True)
df = results['Delta_f']
ddf = results['dDelta_f']
print(f"[BAR:pymbar] ΔF = {df:.8f} ± {ddf:.8f} (1σ)")
print(f"[BAR:pymbar] 95% CI = [{df-1.96*ddf:.8f}, {df+1.96*ddf:.8f}] width={2*1.96*ddf:.8f}")
