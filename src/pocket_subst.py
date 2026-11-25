"""
pocket_subst.py -- specificity control for the DrugCLIP MaxSim reranker (per advisor
feedback). Core question: is the reranker's gain genuine pocket-ligand complementarity,
or ligand-only signal (decoy bias; Chen et al. 2019)?

For each held-out DUD-E test target we take DrugCLIP's top-K shortlist (by dot, scores
fixed) and rerank it with dot + alpha * z(score), where `score` is one of:
  * maxsim(correct pocket)         -- our method, with the target's own pocket
  * maxsim(decoy pocket)           -- several OTHER test-target pockets matched on
                                      heavy-atom count + element composition
  * ligand-only head               -- a trained global query over ligand atoms, no pocket
  * mean-pooled head               -- cos of mean-pooled projected pocket/ligand atoms
                                      (tests whether the atom-wise MAX matters)
  * dot only                       -- Stage-1 baseline
Read-outs per target: actives in top-50 / top-100 (and additional vs dot), BEDROC on the
K-subset. If correct ~= decoy pocket, the signal is not pocket-specific; if correct >>
decoy, it is. All heads are trained on the SAME 60 DUD-E training targets; layer and
alpha are fixed (layer 12, alpha 0.5) from the DUD-E-validated operating point.

Run (lab PC, env drugclip, from ~/drugclip/DrugCLIP):
  python pocket_subst.py --user-dir ./unimol ./data --valid-subset test --task drugclip \
    --loss in_batch_softmax --arch drugclip --seed 1 --path ../checkpoint_best.pt \
    --max-pocket-atoms 511 --batch-size 8 --num-workers 2 --ddp-backend=c10d \
    --dude-root ~/drugclip/dude_data/data/protein/DUD-E/raw/all \
    --layer 12 --alpha 0.5 --topk 1000 --n-decoy-pockets 3 \
    --dump ~/pocket_subst.json 2>&1 | tee ~/pocket_subst.log ; touch ~/pocket_subst.DONE
"""
import os, sys, json, pickle, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore

from rerank_eval import enc, atom_mask_from_pad, encode_target, get_ds

BEDROC_ALPHA = 80.5
cal_metrics = None   # imported lazily inside main() after --user-dir is loaded


# ---------- pocket element composition (for size/composition-matched decoys) ----------
def element_of(atom_name):
    a = atom_name.strip()
    return a[1] if (a and a[0].isdigit()) else a[0]


def pocket_profile(pocket_lmdb):
    """Return (n_heavy_atoms, element-fraction dict) from the stored pocket, dropping H
    to match the loader's RemoveHydrogenPocketDataset."""
    import lmdb
    env = lmdb.open(pocket_lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        d = pickle.loads(txn.get(b"0"))
    env.close()
    names = [n for n in d["pocket_atoms"] if element_of(n) != "H"]
    n = len(names)
    comp = {}
    for nm in names:
        e = element_of(nm); comp[e] = comp.get(e, 0) + 1
    frac = {e: c / n for e, c in comp.items()} if n else {}
    return n, frac


def encode_pocket_only(task, model, root, target, use_cuda, layer):
    """Encode ONLY the pocket per-atom embeddings (cheap; avoids the full mol loop,
    which for a LIT-PCBA target is ~1.5h). Used to pre-cache decoy pockets."""
    poc, _mol = get_ds(task, root, target)
    ps = next(iter(torch.utils.data.DataLoader(poc, 1, collate_fn=poc.collater)))
    if use_cuda:
        ps = unicore.utils.move_to_cuda(ps)
    Pa, Ppad = enc(model.pocket_model, ps, "pocket", layer)
    return Pa[0][atom_mask_from_pad(Ppad)[0]].float().detach()


def profile_dist(pa, pb):
    """Distance between two pocket profiles: normalized size gap + element-fraction L1."""
    na, fa = pa; nb, fb = pb
    size = abs(na - nb) / max(na, nb, 1)
    elems = set(fa) | set(fb)
    comp = sum(abs(fa.get(e, 0.0) - fb.get(e, 0.0)) for e in elems)
    return size + comp


# ---------- scoring heads ----------
def _norm(x):
    return F.normalize(x.float(), dim=-1)


def score_maxsim(l, p, Wm, Wp):
    L = _norm(Wm(l.float())); P = _norm(Wp(p.float()))
    return (L @ P.T).max(dim=1).values.mean()


def score_meanpool(l, p, Wm, Wp):
    # cosine of mean-pooled projected atom embeddings (no atom-wise max)
    lp = F.normalize(Wm(l.float()).mean(dim=0), dim=-1)
    pp = F.normalize(Wp(p.float()).mean(dim=0), dim=-1)
    return (lp * pp).sum()


def score_meanmatch(l, p, Wm, Wp):
    # MATCHED mean control: identical projection and identical PER-ATOM L2 normalization
    # as score_maxsim; differs ONLY in the aggregation over pocket atoms (mean, not max).
    L = _norm(Wm(l.float())); P = _norm(Wp(p.float()))
    return (L @ P.T).mean(dim=1).mean()


def score_ligonly(l, q, Wq):
    # ligand-only: mean cosine of projected ligand atoms to a trained global query (no pocket)
    L = _norm(Wq(l.float()))
    return (L @ F.normalize(q, dim=-1)).mean()


def _z(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-8)


# ---------- training the three heads on shared train-target encodings ----------
def encode_train_cache(args, task, model, use_cuda, layer, full, train_ts):
    cache = {}
    from tqdm import tqdm
    for t in tqdm(train_ts, desc=f"encode train L{layer}"):
        p, acts, decs = encode_target(task, model, args.dude_root, t, use_cuda,
                                      args.decoys_per_target, layer, full)
        if len(acts) > 0 and len(decs) >= args.neg_per_step:
            cache[t] = (p, acts, decs)
    return cache


def train_head(cache, dev, args, kind):
    """kind in {maxsim, meanpool, ligonly}. Returns a dict of trained params."""
    ts = list(cache.keys())
    d = args.proj_dim
    Wm = torch.nn.Linear(512, d, bias=False).to(dev)
    Wp = Wm if args.share_w else torch.nn.Linear(512, d, bias=False).to(dev)
    q = None
    params = list(Wm.parameters()) + ([] if args.share_w else list(Wp.parameters()))
    if kind == "ligonly":
        Wp = None
        q = torch.nn.Parameter(torch.randn(d, device=dev) * 0.1)
        params = list(Wm.parameters()) + [q]
    drop = torch.nn.Dropout(args.dropout)
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=args.proj_wd)
    K, pool, tau = args.neg_per_step, args.hard_pool, args.tau

    def sc(l, p):
        if kind == "maxsim":
            return score_maxsim(l, p, Wm, Wp)
        if kind == "meanpool":
            return score_meanpool(l, p, Wm, Wp)
        if kind == "meanmatch":
            return score_meanmatch(l, p, Wm, Wp)
        return score_ligonly(l, q, Wm)

    for m in [Wm] + ([Wp] if (Wp is not None and not args.share_w) else []):
        m.train()
    for step in range(args.steps):
        t = random.choice(ts)
        p, acts, decs = cache[t]
        a = random.choice(acts)
        idx = random.sample(range(len(decs)), min(pool, len(decs)))
        with torch.no_grad():
            ps = torch.stack([sc(decs[j], p) for j in idx])
        hard = [idx[j] for j in ps.topk(min(K, len(idx))).indices.tolist()]
        s = [sc(a, p)] + [sc(decs[j], p) for j in hard]
        loss = -F.log_softmax(torch.stack(s) / tau, dim=0)[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  [{kind} L{args.layer}] step {step+1:5d}  loss {loss.item():.4f}", flush=True)
    for m in [Wm] + ([Wp] if (Wp is not None and not args.share_w) else []):
        m.eval()
    return {"Wm": Wm, "Wp": Wp, "q": q}


# ---------- metrics on a shortlist ----------
def actives_in_topk(scores, labels, k):
    order = np.argsort(-np.asarray(scores), kind="mergesort")[:k]
    return int(np.asarray(labels)[order].sum())


def bedroc_of(scores, labels):
    auc, bed, ef, _ = cal_metrics(np.asarray(labels), np.asarray(scores, dtype=float), BEDROC_ALPHA)
    return bed, ef["0.005"], ef["0.01"]


def boot_delta(deltas, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi, ("*" if (lo > 0 or hi < 0) else " ")


def main(args):
    global cal_metrics
    from unimol.tasks.drugclip import cal_metrics as _cm
    cal_metrics = _cm
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args); model.load_state_dict(state["model"], strict=False)
    if use_cuda:
        model.cuda()
    model.eval()
    dev = next(model.parameters()).device
    full = len(model.mol_model.encoder.layers)
    layer = args.layer

    targets = sorted(os.listdir(args.dude_root))
    train_ts = targets[args.n_test + args.n_val: args.n_test + args.n_val + args.n_train]
    # test targets: DUD-E held-out by default, or an external root (e.g. LIT-PCBA)
    test_root = args.test_root or args.dude_root
    partner = {}   # cognate -> related off-target pocket, when --pairs is given
    if args.pairs:
        for pr in args.pairs.split(","):
            a, b = pr.split(":")
            partner[a.strip()] = b.strip(); partner[b.strip()] = a.strip()
        test_ts = [t for t in sorted(partner)
                   if os.path.exists(os.path.join(test_root, t, "pocket.lmdb"))
                   and os.path.exists(os.path.join(test_root, t, "mols.lmdb"))]
        held_out = set(targets[:args.n_test]) | set(targets[args.n_test:args.n_test+args.n_val]) \
                   | set(targets[args.n_test+args.n_val+args.n_train:])
        print(f"[pairs/selectivity] decoy pocket = related partner; "
              f"held-out-from-W-training targets: {sorted(t for t in test_ts if t in held_out)}")
    elif args.test_root:
        test_ts = [t for t in sorted(os.listdir(args.test_root))
                   if os.path.exists(os.path.join(args.test_root, t, "pocket.lmdb"))
                   and os.path.exists(os.path.join(args.test_root, t, "mols.lmdb"))]
    else:
        test_ts = targets[: args.n_test]
    print(f"test root: {test_root}")
    print(f"test targets ({len(test_ts)}): {test_ts}")
    print(f"train targets: {len(train_ts)} | layer {layer} | alpha {args.alpha} | topK {args.topk}")

    # 1) train the three heads on the shared train cache
    cache = encode_train_cache(args, task, model, use_cuda, layer, full, train_ts)
    heads = {}
    for kind in ["maxsim", "meanpool", "meanmatch", "ligonly"]:
        print(f"[train] {kind}", flush=True)
        heads[kind] = train_head(cache, dev, args, kind)
    for t in list(cache.keys()):
        del cache[t]
    torch.cuda.empty_cache()

    # 2) pocket profiles for decoy-pocket matching (test pockets only, all held out)
    profiles = {}
    for t in test_ts:
        profiles[t] = pocket_profile(os.path.join(test_root, t, "pocket.lmdb"))

    def decoy_pockets_for(t):
        if partner:   # selectivity mode: the single "decoy" is the related off-target
            return [partner[t]] if partner.get(t) in test_ts else []
        others = [u for u in test_ts if u != t]
        others.sort(key=lambda u: profile_dist(profiles[t], profiles[u]))
        return others[: args.n_decoy_pockets]

    # pre-encode ALL test pockets (cheap, pocket-only) so decoy lookups never re-encode mols
    pocket_atoms_cache = {}
    for t in test_ts:
        pocket_atoms_cache[t] = encode_pocket_only(task, model, test_root, t, use_cuda, layer)

    # 3) per-target: encode mols once, rank by dot, rerank top-K with each head/pocket
    rows = {}
    for t in test_ts:
        p_atoms, pcls, atoms, labels, clss = encode_target(
            task, model, test_root, t, use_cuda, None, layer, full)
        labels = np.asarray(labels)
        dot = (F.normalize(torch.stack(clss), dim=-1) @ pcls.T).squeeze(1).cpu().numpy()
        k = min(args.topk, len(dot))
        top = np.argsort(-dot, kind="mergesort")[:k]
        sub_dot = dot[top]; sub_lab = labels[top]
        sub_atoms = [atoms[i] for i in top]
        n_act_shortlist = int(sub_lab.sum())

        Wm, Wp = heads["maxsim"]["Wm"], heads["maxsim"]["Wp"]
        Wmm, Wmp = heads["meanpool"]["Wm"], heads["meanpool"]["Wp"]
        Wx, Wxp = heads["meanmatch"]["Wm"], heads["meanmatch"]["Wp"]
        Wlq, q = heads["ligonly"]["Wm"], heads["ligonly"]["q"]

        # correct-pocket maxsim, meanpool, meanmatch, ligonly for the K shortlist
        ms_correct = np.array([score_maxsim(a.to(dev), p_atoms, Wm, Wp).item() for a in sub_atoms])
        mp_correct = np.array([score_meanpool(a.to(dev), p_atoms, Wmm, Wmp).item() for a in sub_atoms])
        mx_correct = np.array([score_meanmatch(a.to(dev), p_atoms, Wx, Wxp).item() for a in sub_atoms])
        lo_score = np.array([score_ligonly(a.to(dev), q, Wlq).item() for a in sub_atoms])

        # decoy pockets: maxsim with other targets' pockets (same W, alpha, normalization)
        dps = decoy_pockets_for(t)
        ms_decoys = []
        for u in dps:
            pu = pocket_atoms_cache[u]
            ms_decoys.append(np.array([score_maxsim(a.to(dev), pu, Wm, Wp).item() for a in sub_atoms]))

        def rerank(sc):
            return _z(sub_dot) + args.alpha * _z(sc)

        cand = {
            "dot": sub_dot,
            "correct": rerank(ms_correct),
            "meanpool": rerank(mp_correct),
            "meanmatch": rerank(mx_correct),
            "ligonly": rerank(lo_score),
        }
        for i, u in enumerate(dps):
            cand[f"decoy_{u}"] = rerank(ms_decoys[i])
        # mean over decoy pockets (a single "wrong-pocket" summary)
        if ms_decoys:
            cand["decoy_mean"] = rerank(np.mean(ms_decoys, axis=0))

        rec = {"n_act_shortlist": n_act_shortlist, "k": int(k),
               "a50": {}, "a100": {}, "bedroc": {}}
        for name, sc in cand.items():
            rec["a50"][name] = actives_in_topk(sc, sub_lab, 50)
            rec["a100"][name] = actives_in_topk(sc, sub_lab, 100)
            rec["bedroc"][name] = bedroc_of(sc, sub_lab)[0]
        rows[t] = rec
        da50 = rec["a50"]["correct"] - rec["a50"]["dot"]
        dd50 = rec["a50"].get("decoy_mean", rec["a50"]["dot"]) - rec["a50"]["dot"]
        print(f"  {t:7s} shortlist_act {n_act_shortlist:4d} | a50 dot {rec['a50']['dot']:3d} "
              f"correct {rec['a50']['correct']:3d} (Δ{da50:+d}) decoy {rec['a50'].get('decoy_mean','-')} "
              f"(Δ{dd50:+d}) ligonly {rec['a50']['ligonly']:3d} meanpool {rec['a50']['meanpool']:3d}",
              flush=True)
        torch.cuda.empty_cache()

    # 4) aggregate
    names = list(test_ts)
    def col(metric, key):
        return np.array([rows[t][metric][key] for t in names], dtype=float)

    print("\n===== aggregate over %d test targets (top-%d shortlist) =====" % (len(names), args.topk))
    print("shortlist actives (mean): %.1f" % np.mean([rows[t]["n_act_shortlist"] for t in names]))
    header = f"{'head':10s} {'a50(mean)':>10s} {'Δa50 vs dot':>22s} {'a100(mean)':>10s} {'BEDROC':>8s}"
    print(header); print("-" * len(header))
    keys = ["dot", "correct", "decoy_mean", "ligonly", "meanpool", "meanmatch"]
    for key in keys:
        if not all(key in rows[t]["a50"] for t in names):
            continue
        a50 = col("a50", key); a100 = col("a100", key); bed = col("bedroc", key)
        if key == "dot":
            print(f"{key:10s} {a50.mean():10.2f} {'(baseline)':>22s} {a100.mean():10.2f} {bed.mean():8.4f}")
        else:
            m, lo, hi, s = boot_delta(a50 - col("a50", "dot"))
            print(f"{key:10s} {a50.mean():10.2f} {m:+6.2f}[{lo:+.2f},{hi:+.2f}]{s} {a100.mean():10.2f} {bed.mean():8.4f}")
    # the key contrast: correct vs decoy pocket (paired, per target)
    if all("decoy_mean" in rows[t]["a50"] for t in names):
        for metric, lab in [("a50", "actives@50"), ("a100", "actives@100"), ("bedroc", "BEDROC")]:
            m, lo, hi, s = boot_delta(col(metric, "correct") - col(metric, "decoy_mean"))
            print(f"  correct - decoy  {lab:12s} Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] {s}")
        for metric, lab in [("a50", "actives@50"), ("bedroc", "BEDROC")]:
            m, lo, hi, s = boot_delta(col(metric, "correct") - col(metric, "ligonly"))
            print(f"  correct - ligonly {lab:11s} Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] {s}")
            m, lo, hi, s = boot_delta(col(metric, "correct") - col(metric, "meanpool"))
            print(f"  correct - meanpool {lab:10s} Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] {s}")
            m, lo, hi, s = boot_delta(col(metric, "correct") - col(metric, "meanmatch"))
            print(f"  correct - meanmatch {lab:9s} Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] {s}  (matched-norm mean vs max)")

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({"rows": rows, "test_ts": names,
                       "decoy_pockets": {t: decoy_pockets_for(t) for t in names},
                       "layer": layer, "alpha": args.alpha, "topk": args.topk}, f)
        print(f"\n[dumped per-target pocket-substitution results to {args.dump}]")


def cli():
    p = options.get_validation_parser()
    p.add_argument("--dude-root", type=str, required=True)
    p.add_argument("--test-root", type=str, default="",
                   help="external test root (e.g. LIT-PCBA built dir); train still on DUD-E")
    p.add_argument("--pairs", type=str, default="",
                   help="selectivity mode: comma list of related DUD-E pairs 'A:B,C:D'; "
                        "for each target the single decoy pocket is its related partner")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--n-decoy-pockets", type=int, default=3)
    p.add_argument("--dump", type=str, default="")
    p.add_argument("--n-train", type=int, default=60)
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--n-test", type=int, default=20)
    p.add_argument("--decoys-per-target", type=int, default=800)
    p.add_argument("--neg-per-step", type=int, default=32)
    p.add_argument("--hard-pool", type=int, default=96)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--proj-wd", type=float, default=1e-2)
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--share-w", action="store_true")
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
