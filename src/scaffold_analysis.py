"""
scaffold_analysis.py -- deployment utility table (advisor item 2), CPU-only.

For each held-out DUD-E test target, reconstruct the DrugCLIP-dot ranking and the
reranked ordering (dot + 0.5*z(maxsim)) from the dumped per-mol scores, then report:
  * actives in the top-50 / top-100 (dot vs reranked, and the ADDITIONAL actives)
  * distinct active Bemis-Murcko scaffolds in the top-50 / top-100 (dot vs reranked)
  * actives already in the Stage-1 shortlist (top-1000 by dot) -- the reranker ceiling
SMILES come from each target's mols.lmdb (index-aligned to the score dump); scaffolds
via RDKit. No GPU.

Run (lab PC, env drugclip): python scaffold_analysis.py ~/dude_scores.pkl \
  --dude-root ~/drugclip/dude_data/data/protein/DUD-E/raw/all --layer 12 --alpha 0.5
"""
import os, sys, pickle, argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def z(x):
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-8)


def scaffold_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return None


def read_smiles(lmdb_path, indices):
    import lmdb
    env = lmdb.open(lmdb_path, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    out = {}
    with env.begin() as txn:
        for i in indices:
            v = txn.get(str(int(i)).encode())
            d = pickle.loads(v)
            out[i] = (d.get("smi"), int(d.get("label", -1)))
    env.close()
    return out


def distinct_active_scaffolds(order_k, labels, smi_map):
    scafs = set()
    for i in order_k:
        if labels[i] == 1:
            s = smi_map.get(i, (None, None))[0]
            sc = scaffold_smiles(s) if s else None
            if sc:
                scafs.add(sc)
    return len(scafs)


def boot_delta(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi, ("*" if (lo > 0 or hi < 0) else " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("--dude-root", required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--shortlist", type=int, default=1000)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    S = pickle.load(open(args.pickle, "rb"))
    meta = S.get("__meta__", {})
    test = meta.get("test") or sorted({k.split("|")[1] for k in S if k != "__meta__"})
    test = [t for t in test if f"{args.layer}|{t}" in S]
    print(f"deployment utility on {len(test)} DUD-E test targets | layer {args.layer} | "
          f"rerank dot+{args.alpha:g}*z(maxsim)\n")

    hdr = (f"{'target':8s} {'shortl.act':>10s} | {'a50 dot/rr(+)':>14s} {'scaf50 dot/rr':>13s} | "
           f"{'a100 dot/rr(+)':>15s} {'scaf100 dot/rr':>14s}")
    print(hdr); print("-" * len(hdr))

    rows = {}
    for t in test:
        d = S[f"{args.layer}|{t}"]
        dot = np.asarray(d["dot"], float); ms = np.asarray(d["maxsim"], float)
        lab = np.asarray(d["labels"]).astype(int)
        comb = z(dot) + args.alpha * z(ms)
        o_dot = np.argsort(-dot, kind="mergesort")
        o_rr = np.argsort(-comb, kind="mergesort")
        shortlist_act = int(lab[o_dot[:args.shortlist]].sum())

        need = set(o_dot[:100]).union(o_rr[:100])
        smi_map = read_smiles(os.path.join(args.dude_root, t, "mols.lmdb"), need)

        rec = {"shortlist_act": shortlist_act, "total_act": int(lab.sum())}
        for k in (50, 100):
            rec[f"a{k}_dot"] = int(lab[o_dot[:k]].sum())
            rec[f"a{k}_rr"] = int(lab[o_rr[:k]].sum())
            rec[f"s{k}_dot"] = distinct_active_scaffolds(o_dot[:k], lab, smi_map)
            rec[f"s{k}_rr"] = distinct_active_scaffolds(o_rr[:k], lab, smi_map)
        rows[t] = rec
        print(f"{t:8s} {shortlist_act:10d} | "
              f"{rec['a50_dot']:3d}/{rec['a50_rr']:3d}(+{rec['a50_rr']-rec['a50_dot']:d})".rjust(14)
              + f" {rec['s50_dot']:5d}/{rec['s50_rr']:<5d}".rjust(13) + " | "
              + f"{rec['a100_dot']:3d}/{rec['a100_rr']:3d}(+{rec['a100_rr']-rec['a100_dot']:d})".rjust(15)
              + f" {rec['s100_dot']:5d}/{rec['s100_rr']:<5d}".rjust(14))

    # aggregate
    def col(key):
        return np.array([rows[t][key] for t in test], float)
    print("\n===== aggregate (mean over %d targets) =====" % len(test))
    print(f"shortlist actives (top-{args.shortlist} by dot): {col('shortlist_act').mean():.1f}  "
          f"(total actives/target {col('total_act').mean():.0f})")
    for k in (50, 100):
        m, lo, hi, s = boot_delta(col(f"a{k}_rr") - col(f"a{k}_dot"))
        sm, slo, shi, ss = boot_delta(col(f"s{k}_rr") - col(f"s{k}_dot"))
        print(f"top-{k:3d}: actives dot {col(f'a{k}_dot').mean():5.2f} -> rerank {col(f'a{k}_rr').mean():5.2f} "
              f"(Δ {m:+.2f} [{lo:+.2f},{hi:+.2f}]{s}) | "
              f"distinct active scaffolds dot {col(f's{k}_dot').mean():5.2f} -> rerank {col(f's{k}_rr').mean():5.2f} "
              f"(Δ {sm:+.2f} [{slo:+.2f},{shi:+.2f}]{ss})")

    if args.dump:
        import json
        json.dump({"rows": rows, "targets": test}, open(args.dump, "w"))
        print("dumped per-target to", args.dump)


if __name__ == "__main__":
    main()
