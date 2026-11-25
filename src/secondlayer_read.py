"""
secondlayer_read.py -- offline read of the --with-raw --dump-metrics JSON to build
the DrugCLIP "second-layer" baseline comparison on the 20 held-out DUD-E test targets.

DrugCLIP's own variant adds RAW second-to-last-layer atom embeddings to the global
score (Science App C.2): last layer hurt screening, 2nd-to-last was ~neutral. Our
"dot+0.5raw" at each layer IS that untrained baseline; "dot+0.5" is the same with a
trained projection W. This tabulates, per layer, dot vs dot+0.5raw (DrugCLIP-style)
vs dot+0.5 (ours), with paired bootstrap Δ vs dot over the test targets.

Usage:  python secondlayer_read.py ~/secondlayer.json
"""
import argparse, json
import numpy as np

TEST = ["aa2ar", "abl1", "ace", "aces", "ada", "ada17", "adrb1", "adrb2", "akt1",
        "akt2", "aldr", "ampc", "andr", "aofb", "bace1", "braf", "cah2", "casp3",
        "cdk2", "comt"]
BEDROC, EF005 = 1, 2


def boot_delta(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi, ("*" if (lo > 0 or hi < 0) else " "), int((d > 0).sum()), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    args = ap.parse_args()
    P = json.load(open(args.json))
    layers = sorted({int(k.split("|")[0]) for k in P}, reverse=True)
    test = [t for t in TEST if all(f"{L}|{t}" in P for L in layers)]
    print(f"second-layer baseline on {len(test)} held-out DUD-E test targets\n")
    for metric, name in [(BEDROC, "BEDROC"), (EF005, "EF@0.5%")]:
        print(f"===== {name} =====")
        print(f"{'layer':>5s} {'dot':>7s} {'dot+0.5raw':>11s} {'Δraw vs dot (DrugCLIP-style)':>34s}"
              f" {'dot+0.5(W)':>11s} {'ΔW vs dot (ours)':>26s}")
        for L in layers:
            dot = np.array([P[f"{L}|{t}"]["dot"][metric] for t in test])
            raw = np.array([P[f"{L}|{t}"]["dot+0.5raw"][metric] for t in test])
            trn = np.array([P[f"{L}|{t}"]["dot+0.5"][metric] for t in test])
            mr, lor, hir, sr, wr, n = boot_delta(raw - dot)
            mw, low, hiw, sw, ww, _ = boot_delta(trn - dot)
            print(f"{L:5d} {dot.mean():7.3f} {raw.mean():11.3f} "
                  f"{mr:+.3f}[{lor:+.3f},{hir:+.3f}]{sr}{wr:>2d}/{n} "
                  f"{trn.mean():11.3f} {mw:+.3f}[{low:+.3f},{hiw:+.3f}]{sw}{ww:>2d}/{n}")
        print()
    print("* = paired bootstrap 95% CI over targets excludes 0")
    print("dot+0.5raw = DrugCLIP-style untrained atom embeddings added to the dot;")
    print("dot+0.5(W) = same combination with the trained projection (ours).")


if __name__ == "__main__":
    main()
