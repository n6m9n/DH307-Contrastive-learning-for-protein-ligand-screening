"""
topk_analyze.py -- offline top-K DEPLOYMENT-CASCADE sweep from the per-mol scores
dumped by rerank_eval.py (--dump-scores). No GPU; imports the SAME cal_metrics the
main experiment uses, so BEDROC/EF are computed identically.

Deployment regime (what the cascade actually does): DrugCLIP's dot ranks the whole
library; keep the top-K; rerank ONLY those K by the combined score (z(dot)+a*z(maxsim));
molecules below rank K keep their dot order. We then score enrichment on that final
ordering -- i.e. Stage-2 can reorder within the shortlist but cannot pull an active
from below K into the top (the Recall@K ceiling). K=all reproduces full reranking;
the dot baseline is the Stage-1 ordering.

Usage (lab PC, env drugclip, from ~/drugclip/DrugCLIP):
  python topk_analyze.py ~/dude_scores.pkl --layer 12 --alpha 0.5
  python topk_analyze.py ~/litpcba_scores.pkl --layer 12 --alpha 1.0
"""
import argparse, pickle
import numpy as np
from unimol.tasks.drugclip import cal_metrics   # identical metric path as rerank_eval

KS = [50, 100, 200, 500, 1000, 2000, 0]          # 0 == all (full rerank)
BEDROC_ALPHA = 80.5


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def deploy_score(dot, maxsim, K, alpha):
    """Final per-mol score whose descending argsort IS the deployment ranking:
    top-K by dot reordered by combined; the rest kept in dot order below them."""
    n = len(dot)
    combined = z(dot) + alpha * z(maxsim)
    order_dot = np.argsort(-dot, kind="mergesort")
    k = n if (K == 0 or K >= n) else K
    topk = order_dot[:k]
    rest = order_dot[k:]
    topk_sorted = topk[np.argsort(-combined[topk], kind="mergesort")]
    final_order = np.concatenate([topk_sorted, rest])
    score = np.empty(n, dtype=np.float64)
    score[final_order] = np.arange(n, 0, -1)     # rank-1 gets the highest score
    return score


def recall_at_k(dot, labels, K):
    n = len(dot)
    k = n if (K == 0 or K >= n) else K
    order = np.argsort(-dot, kind="mergesort")[:k]
    pos = int((labels == 1).sum())
    return float((labels[order] == 1).sum()) / pos if pos else float("nan")


def boot_delta(deltas, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    sig = "*" if (lo > 0 or hi < 0) else " "
    return d.mean(), lo, hi, sig, int((d > 0).sum()), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--targets", default="", help="comma list; default = dumped test set")
    args = ap.parse_args()

    S = pickle.load(open(args.pickle, "rb"))
    meta = S.get("__meta__", {})
    targets = ([t.strip() for t in args.targets.split(",") if t.strip()]
               or meta.get("test") or
               sorted({k.split("|")[1] for k in S if k != "__meta__"}))
    targets = [t for t in targets if f"{args.layer}|{t}" in S]
    print(f"top-K deployment cascade | layer {args.layer} | alpha {args.alpha} | "
          f"{len(targets)} targets")
    print(f"dumped from: {meta.get('test_root','?')}")

    # per-target raw arrays
    data = {t: S[f"{args.layer}|{t}"] for t in targets}
    sizes = {t: len(data[t]["dot"]) for t in targets}
    print(f"library sizes: min {min(sizes.values())}, median "
          f"{int(np.median(list(sizes.values())))}, max {max(sizes.values())}\n")

    # dot baseline per target (K irrelevant): bedroc, ef0.5, ef1
    base = {}
    for t in targets:
        d = data[t]
        auc, bed, ef, _ = cal_metrics(d["labels"], d["dot"].astype(float), BEDROC_ALPHA)
        base[t] = (bed, ef["0.005"], ef["0.01"])
    b_bed = np.mean([base[t][0] for t in targets])
    b_e5 = np.mean([base[t][1] for t in targets])
    b_e1 = np.mean([base[t][2] for t in targets])
    print(f"{'DrugCLIP dot (Stage-1 baseline)':34s}  BEDROC {b_bed:.4f}   "
          f"EF@0.5% {b_e5:6.2f}   EF@1% {b_e1:6.2f}\n")

    hdr = (f"{'K':>6s} {'Recall@K':>9s} | {'BEDROC':>7s} {'ΔBEDROC (95% CI) win':>30s} | "
           f"{'EF0.5%':>7s} {'ΔEF0.5% (95% CI) win':>30s} | {'EF1%':>7s} {'ΔEF1%':>8s}")
    print(hdr); print("-" * len(hdr))
    for K in KS:
        rr_bed, rr_e5, rr_e1, rec = [], [], [], []
        d_bed, d_e5, d_e1 = [], [], []
        for t in targets:
            d = data[t]
            sc = deploy_score(d["dot"].astype(float), d["maxsim"].astype(float), K, args.alpha)
            auc, bed, ef, _ = cal_metrics(d["labels"], sc, BEDROC_ALPHA)
            rr_bed.append(bed); rr_e5.append(ef["0.005"]); rr_e1.append(ef["0.01"])
            rec.append(recall_at_k(d["dot"].astype(float), d["labels"], K))
            d_bed.append(bed - base[t][0]); d_e5.append(ef["0.005"] - base[t][1])
            d_e1.append(ef["0.01"] - base[t][2])
        mb, lob, hib, sb, wb, n = boot_delta(d_bed)
        m5, lo5, hi5, s5, w5, _ = boot_delta(d_e5)
        m1, lo1, hi1, s1, w1, _ = boot_delta(d_e1)
        klab = "all" if K == 0 else str(K)
        print(f"{klab:>6s} {np.mean(rec):9.3f} | {np.mean(rr_bed):7.4f} "
              f"{mb:+.4f}[{lob:+.4f},{hib:+.4f}]{sb}{wb:>2d}/{n} | "
              f"{np.mean(rr_e5):7.2f} {m5:+.2f}[{lo5:+.2f},{hi5:+.2f}]{s5}{w5:>2d}/{n} | "
              f"{np.mean(rr_e1):7.2f} {m1:+.2f}{s1}")
    print("\n* = bootstrap 95% CI (paired per-target Δ vs dot, over targets) excludes 0")
    print("Recall@K is the Stage-1 ceiling: actives below rank K cannot be promoted.")


if __name__ == "__main__":
    main()
