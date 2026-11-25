"""
loo_analyze.py -- offline leave-one-target-out (LOO) analysis of the LIT-PCBA
transfer, from the per-(layer,target) metrics dumped by rerank_eval.py
(--dump-metrics). No GPU, no numpy; pure python.

For each held-out target: select the operating point (layer x alpha) by mean
BEDROC on the OTHER 14 targets, then record the held-out target's improvement
over dot. Bootstrap over the 15 held-out deltas -> tight, split-free estimate.

Usage:
  python loo_analyze.py ~/litpcba_permetrics.json
  python loo_analyze.py ~/litpcba_permetrics.json --targets ADRB2,ALDH1,...
"""
import argparse
import json
import random

# the 15 LIT-PCBA targets (names as built under litpcba_data/)
LITPCBA = ["ADRB2", "ALDH1", "ESR1_ago", "ESR1_ant", "FEN1", "GBA", "IDH1",
           "KAT2A", "MAPK1", "MTORC1", "OPRK1", "PKM2", "PPARG", "TP53", "VDR"]
SEL_OPS = ["dot+0.25", "dot+0.5", "dot+1"]   # learned-combo candidates to select among
BEDROC, EF005 = 1, 2                          # indices into [auc, bedroc, ef0.5, ef1]


def boot_ci(deltas, n_boot=10000, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return sum(deltas) / n, lo, hi, sum(1 for d in deltas if d > 0), n


def cell(P, layer, t):
    return P[f"{layer}|{t}"]


def loo(P, targets, layers, metric):
    """Leave-one-out: select (layer, op) on the other 14 by mean BEDROC, record
    held-out delta of the selected op vs dot at that layer, for `metric`."""
    deltas, picks = [], []
    for held in targets:
        others = [t for t in targets if t != held]
        best = None  # (layer, op, mean_bedroc_on_others)
        for L in layers:
            for op in SEL_OPS:
                mb = sum(cell(P, L, t)[op][BEDROC] for t in others) / len(others)
                if best is None or mb > best[2]:
                    best = (L, op, mb)
        L, op, _ = best
        d = cell(P, L, held)[op][metric] - cell(P, L, held)["dot"][metric]
        deltas.append(d)
        picks.append((L, op))
    return deltas, picks


def full_pool_best(P, targets, layers):
    best = None
    for L in layers:
        for op in SEL_OPS:
            mb = sum(cell(P, L, t)[op][BEDROC] for t in targets) / len(targets)
            if best is None or mb > best[2]:
                best = (L, op, mb)
    return best[0], best[1]


def report(tag, deltas, label):
    m, lo, hi, wins, n = boot_ci(deltas)
    sig = " *" if (lo > 0 or hi < 0) else ""
    print(f"  {tag:10s} {label:8s}  mean Δ {m:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]  "
          f"win {wins}/{n}{sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--targets", default=",".join(LITPCBA))
    args = ap.parse_args()
    P = json.load(open(args.json))
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    layers = sorted({int(k.split("|")[0]) for k in P}, reverse=True)
    # keep only targets that have all layers present
    targets = [t for t in targets if all(f"{L}|{t}" in P for L in layers)]
    print(f"targets (n={len(targets)}): {targets}")
    print(f"layers: {layers}   candidate ops: {SEL_OPS}")

    fl, fo = full_pool_best(P, targets, layers)
    print(f"\nfull-pool best operating point (all {len(targets)}, by BEDROC): layer {fl}, {fo}")

    print("\n===== LEAVE-ONE-OUT CV (operating point re-selected on the other n-1) =====")
    for metric, name in [(BEDROC, "BEDROC"), (EF005, "EF@0.5%")]:
        deltas, picks = loo(P, targets, layers, metric)
        report("LOO", deltas, name)
        if metric == BEDROC:
            from collections import Counter
            pc = Counter(f"L{L}/{op}" for L, op in picks)
            print(f"             selected per fold: {dict(pc)}")

    print("\n===== reference: raw-W ablation (dot+0.5raw − dot), all targets, per layer =====")
    for L in layers:
        if all("dot+0.5raw" in cell(P, L, t) for t in targets):
            db = [cell(P, L, t)["dot+0.5raw"][BEDROC] - cell(P, L, t)["dot"][BEDROC] for t in targets]
            report(f"L{L}", db, "BEDROC")


if __name__ == "__main__":
    main()
