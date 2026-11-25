import os, random, json
import torch, numpy as np, torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore
from tqdm import tqdm


def enc(tower, sample, pfx, n_layers):
    dist = sample["net_input"][f"{pfx}_src_distance"]
    et = sample["net_input"][f"{pfx}_src_edge_type"]
    st = sample["net_input"][f"{pfx}_src_tokens"]
    pad = st.eq(tower.padding_idx)
    emb = tower.embed_tokens(st)
    n = dist.size(-1)
    bias = tower.gbf_proj(tower.gbf(dist, et)).permute(0, 3, 1, 2).contiguous().view(-1, n, n)
    E = tower.encoder
    if n_layers is None or n_layers >= len(E.layers):
        return E(emb, padding_mask=pad, attn_mask=bias)[0], pad
    h = E.emb_layer_norm(emb)
    h = h * (1 - pad.unsqueeze(-1).type_as(h))
    sl = h.size(1)
    am = bias.view(h.size(0), -1, sl, sl).masked_fill(
        pad.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")).view(-1, sl, sl)
    for i in range(n_layers):
        h, am, _ = E.layers[i](h, padding_mask=None, attn_bias=am, return_attn=True)
    return h, pad


def atom_mask_from_pad(pad):
    nonpad = ~pad
    m = nonpad.clone()
    m[:, 0] = False
    lens = nonpad.sum(1)
    m[torch.arange(pad.size(0), device=pad.device), lens - 1] = False
    return m


_DS = {}


def get_ds(task, root, target):
    key = (root, target)
    if key not in _DS:
        base = f"{root}/{target}"
        poc = task.load_pockets_dataset(base + "/pocket.lmdb")
        mol = task.load_mols_dataset(base + "/mols.lmdb", "atoms", "coordinates")
        _DS[key] = (poc, mol)
    return _DS[key]


@torch.no_grad()
def encode_target(task, model, root, target, use_cuda, max_decoys, layer, full):
    poc, mol = get_ds(task, root, target)
    need_cls = max_decoys is None
    ps = next(iter(torch.utils.data.DataLoader(poc, 1, collate_fn=poc.collater)))
    if use_cuda:
        ps = unicore.utils.move_to_cuda(ps)
    Pa, Ppad = enc(model.pocket_model, ps, "pocket", layer)
    p_atoms = Pa[0][atom_mask_from_pad(Ppad)[0]].float()
    pcls = None
    if need_cls:
        Pf = Pa if layer >= full else enc(model.pocket_model, ps, "pocket", None)[0]
        pcls = F.normalize(model.pocket_project(Pf[0, 0:1]).float(), dim=-1)

    actives, decoys, all_atoms, all_labels, all_cls = [], [], [], [], []
    for samp in torch.utils.data.DataLoader(mol, 64, collate_fn=mol.collater):
        if use_cuda:
            samp = unicore.utils.move_to_cuda(samp)
        La, Lpad = enc(model.mol_model, samp, "mol", layer)
        amask = atom_mask_from_pad(Lpad)
        labs = samp["target"].detach().cpu().numpy()
        if need_cls:
            Lf = La if layer >= full else enc(model.mol_model, samp, "mol", None)[0]
            cls = model.mol_project(Lf[:, 0, :]).float()
        for i in range(La.size(0)):
            ai = La[i][amask[i]].float()
            if need_cls:
                # keep per-atom embeddings on CPU (fp16): full LIT-PCBA targets reach
                # ~350k mols; holding all atoms on the 20GB GPU would OOM. Scored one
                # at a time back on the GPU in eval_split.
                all_atoms.append(ai.half().cpu()); all_labels.append(int(labs[i])); all_cls.append(cls[i])
            elif labs[i] == 1:
                actives.append(ai.half())
            elif len(decoys) < max_decoys:
                decoys.append(ai.half())
        if (not need_cls) and len(decoys) >= max_decoys:
            break
    if need_cls:
        return p_atoms, pcls, all_atoms, np.array(all_labels), all_cls
    return p_atoms.half(), actives, decoys


def maxsim(l, p, Wm, Wp, drop=None):
    L = Wm(l.float()); P = Wp(p.float())
    if drop is not None:
        L = drop(L)
    L = F.normalize(L, dim=-1); P = F.normalize(P, dim=-1)
    return (L @ P.T).max(dim=1).values.mean()


def raw_maxsim(l, p):
    L = F.normalize(l.float(), dim=-1); P = F.normalize(p.float(), dim=-1)
    return (L @ P.T).max(dim=1).values.mean()


ALPHAS = [0.25, 0.5, 1.0]
PERTGT = {}   # {"<layer>|<target>": {op: [auc, bedroc, ef0.5, ef1]}} when --dump-metrics
SCORES = {}   # {"<layer>|<target>": {dot, maxsim, labels}} when --dump-scores (top-K sweep offline)


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def train_W(args, task, model, dev, use_cuda, full, layer, train_ts):
    cache = {}
    for t in tqdm(train_ts, desc=f"encode train L{layer}"):
        p, acts, decs = encode_target(task, model, args.dude_root, t, use_cuda,
                                      args.decoys_per_target, layer, full)
        if len(acts) > 0 and len(decs) >= args.neg_per_step:
            cache[t] = (p, acts, decs)
    ts = list(cache.keys())
    d = args.proj_dim
    Wm = torch.nn.Linear(512, d, bias=False).to(dev)
    Wp = Wm if args.share_w else torch.nn.Linear(512, d, bias=False).to(dev)
    drop = torch.nn.Dropout(args.dropout)
    tau = args.tau
    params = list(Wm.parameters()) + ([] if args.share_w else list(Wp.parameters()))
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=args.proj_wd)
    K, pool = args.neg_per_step, args.hard_pool
    Wm.train(); Wp.train()
    for step in range(args.steps):
        t = random.choice(ts)
        p, acts, decs = cache[t]
        a = random.choice(acts)
        if args.random_neg:
            hard = random.sample(range(len(decs)), min(K, len(decs)))
        else:
            idx = random.sample(range(len(decs)), min(pool, len(decs)))
            with torch.no_grad():
                ps = torch.stack([maxsim(decs[j], p, Wm, Wp) for j in idx])
            hard = [idx[j] for j in ps.topk(min(K, len(idx))).indices.tolist()]
        s = [maxsim(a, p, Wm, Wp, drop)] + [maxsim(decs[j], p, Wm, Wp, drop) for j in hard]
        loss = -F.log_softmax(torch.stack(s) / tau, dim=0)[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  [L{layer}] step {step+1:5d}  loss {loss.item():.4f}")
    Wm.eval(); Wp.eval()
    for t in list(cache.keys()):
        del cache[t]
    torch.cuda.empty_cache()
    return Wm, Wp


@torch.no_grad()
def eval_split(args, task, model, use_cuda, full, layer, Wm, Wp, split_ts, cal_metrics, desc, root):
    acc = {}
    for t in tqdm(split_ts, desc=desc):
        p_atoms, pcls, atoms, labels, clss = encode_target(task, model, root, t,
                                                           use_cuda, None, layer, full)
        dev = p_atoms.device
        trW = np.array([maxsim(a.to(dev), p_atoms, Wm, Wp).item() for a in atoms])
        dot = (F.normalize(torch.stack(clss), dim=-1) @ pcls.T).squeeze(1).cpu().numpy()
        out = {"dot": dot, "maxsim": trW}
        if args.dump_scores:
            SCORES[f"{layer}|{t}"] = {"dot": np.asarray(dot, dtype=np.float32),
                                      "maxsim": np.asarray(trW, dtype=np.float32),
                                      "labels": np.asarray(labels, dtype=np.int8)}
        for al in ALPHAS:
            out[f"dot+{al:g}"] = _z(dot) + al * _z(trW)
        if args.with_raw:
            rawm = np.array([raw_maxsim(a.to(dev), p_atoms).item() for a in atoms])
            out["rawmaxsim"] = rawm
            out["dot+0.5raw"] = _z(dot) + 0.5 * _z(rawm)
        for name, sc in out.items():
            auc, bedroc, ef, re = cal_metrics(labels, sc, 80.5)
            acc.setdefault(name, []).append((auc, bedroc, ef["0.005"], ef["0.01"]))
            if args.dump_metrics:
                PERTGT.setdefault(f"{layer}|{t}", {})[name] = [
                    float(auc), float(bedroc), float(ef["0.005"]), float(ef["0.01"])]
        del atoms
        torch.cuda.empty_cache()
    return acc


def mean_row(acc, name):
    return np.array(acc[name]).mean(0)


def order_names(args):
    if args.with_raw:
        return ["dot", "maxsim", "rawmaxsim"] + [f"dot+{al:g}" for al in ALPHAS] + ["dot+0.5raw"]
    return ["dot", "maxsim"] + [f"dot+{al:g}" for al in ALPHAS]


def print_table(title, acc, order):
    print(f"\n===== {title}: mean over {len(acc['dot'])} targets =====")
    print(f"{'method':12s}  {'AUC':>13s}  {'BEDROC':>13s}  {'EF@0.5%':>13s}  {'EF@1%':>13s}")
    for name in order:
        A = np.array(acc[name]); m, s = A.mean(0), A.std(0)
        print(f"{name:12s}  {m[0]:6.4f}±{s[0]:5.3f}  {m[1]:6.4f}±{s[1]:5.3f}  "
              f"{m[2]:6.2f}±{s[2]:5.2f}  {m[3]:6.2f}±{s[3]:5.2f}")


def paired(acc, cand, idx, label):
    b = np.array(acc["dot"])[:, idx]
    c = np.array(acc[cand])[:, idx]
    delta = c - b
    n = len(delta); wins = int((delta > 0).sum())
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(delta, n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    sig = "  *" if (lo > 0 or hi < 0) else ""
    print(f"  {label:8s} {cand:9s}- dot:  mean Δ {delta.mean():+.4f}  "
          f"[95% CI {lo:+.4f}, {hi:+.4f}]  win {wins}/{n}{sig}")


def per_target(acc, cand, names):
    b = np.array(acc["dot"]); c = np.array(acc[cand])
    print(f"\n--- per-target (TEST): dot vs {cand} ---")
    print(f"{'target':>10s}  {'dot BEDROC':>10s}  {'sel BEDROC':>10s}  {'ΔBEDROC':>8s}  {'ΔEF0.5%':>8s}")
    for i, nm in enumerate(names):
        print(f"{nm:>10s}  {b[i,1]:10.4f}  {c[i,1]:10.4f}  {c[i,1]-b[i,1]:+8.4f}  {c[i,2]-b[i,2]:+8.2f}")


def main(args):
    from unimol.tasks.drugclip import cal_metrics
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

    # train W + select layer on DUD-E exactly as the validated main experiment
    # (val = targets[20:30], train = targets[30:90]); only the held-out TEST set
    # changes when --test-root points at an external dataset (e.g. LIT-PCBA).
    targets = sorted(os.listdir(args.dude_root))
    train_ts = targets[args.n_test + args.n_val: args.n_test + args.n_val + args.n_train]
    val_root = args.dude_root
    if args.test_root and args.select_on_test_root:
        # FAIR operating-point check: W still trained on DUD-E, but the operating
        # point (layer + alpha) is selected on a held-out LIT-PCBA val split and
        # reported on a disjoint LIT-PCBA test split (deterministic interleave).
        ltp = sorted(d for d in os.listdir(args.test_root)
                     if os.path.isdir(os.path.join(args.test_root, d)))
        val_ts, test_ts = ltp[::2], ltp[1::2]
        val_root = test_root = args.test_root
        mode = f"TRANSFER + in-dataset select: W on DUD-E; select/test split on {args.test_root}"
    elif args.test_root:
        val_ts = targets[args.n_test: args.n_test + args.n_val]
        test_root = args.test_root
        test_ts = sorted(d for d in os.listdir(args.test_root)
                         if os.path.isdir(os.path.join(args.test_root, d)))
        mode = f"TRANSFER: train/val on DUD-E, test on {args.test_root}"
    else:
        val_ts = targets[args.n_test: args.n_test + args.n_val]
        test_root = args.dude_root
        test_ts = targets[: args.n_test]
        mode = "in-domain: train/val/test all on DUD-E"
    layers = [int(x) for x in args.layers.split(",")] if args.layers else [args.layer]
    print(f"encoder layers={full} | train={len(train_ts)} val={len(val_ts)} test={len(test_ts)} | layer sweep: {layers}")
    print(mode)
    print("train:", train_ts)
    print("test :", test_ts)

    # ---- sweep layers: train W, evaluate on val + test, select layer on validation ----
    results = {}   # layer -> (val_bedroc_dot05, val_acc, test_acc)
    for layer in layers:
        Wm, Wp = train_W(args, task, model, dev, use_cuda, full, layer, train_ts)
        val_acc = eval_split(args, task, model, use_cuda, full, layer, Wm, Wp, val_ts, cal_metrics, f"VAL  L{layer}", val_root)
        test_acc = eval_split(args, task, model, use_cuda, full, layer, Wm, Wp, test_ts, cal_metrics, f"TEST L{layer}", test_root)
        vb = mean_row(val_acc, "dot+0.5")[1]
        results[layer] = (vb, val_acc, test_acc)
        print(f"  [layer {layer}] VAL dot+0.5 BEDROC = {vb:.4f}  (val dot = {mean_row(val_acc,'dot')[1]:.4f})")
        del Wm, Wp
        torch.cuda.empty_cache()

    order = order_names(args)
    # select over (layer, alpha) when doing the in-dataset operating-point check;
    # otherwise keep the validated behaviour (fixed op dot+0.5, select layer only).
    sel_ops = [f"dot+{al:g}" for al in ALPHAS] if args.select_on_test_root else ["dot+0.5"]
    print("\n===== OPERATING-POINT SELECTION (on validation, by BEDROC) =====")
    print(f"{'layer':>6s}  {'op':>10s}  {'val BEDROC':>11s}  {'val EF@0.5%':>12s}")
    best = None
    for L in layers:
        _, va, _ = results[L]
        for op in sel_ops:
            vb = mean_row(va, op)[1]
            print(f"{L:6d}  {op:>10s}  {vb:11.4f}  {mean_row(va, op)[2]:12.2f}")
            if best is None or vb > best[2]:
                best = (L, op, vb)
    best_layer, best_op, _ = best
    print(f"\n>>> selected on validation (by BEDROC): layer {best_layer}, operating point {best_op}")

    _, val_acc_b, test_acc_b = results[best_layer]
    print_table(f"VALIDATION (selected layer {best_layer})", val_acc_b, order)
    print_table(f"TEST held-out (selected layer {best_layer})", test_acc_b, order)
    print(f"\n--- paired improvement over dot (TEST, layer {best_layer}, op {best_op}), bootstrap 95% CI, * = CI excludes 0 ---")
    paired(test_acc_b, best_op, 1, "BEDROC")
    paired(test_acc_b, best_op, 2, "EF@0.5%")
    if args.with_raw:
        print("  (learned-W necessity: raw atom cosines in the same combination)")
        paired(test_acc_b, "dot+0.5raw", 1, "BEDROC")
        paired(test_acc_b, "dot+0.5raw", 2, "EF@0.5%")
    per_target(test_acc_b, best_op, test_ts)

    if args.dump_metrics:
        with open(args.dump_metrics, "w") as f:
            json.dump(PERTGT, f)
        print(f"\n[dumped per-target×per-layer metrics for {len(PERTGT)} (layer,target) cells "
              f"to {args.dump_metrics}]")

    if args.dump_scores:
        import pickle
        SCORES["__meta__"] = {"test": list(test_ts), "val": list(val_ts),
                              "layers": layers, "test_root": test_root}
        with open(args.dump_scores, "wb") as f:
            pickle.dump(SCORES, f)
        print(f"\n[dumped per-mol dot+maxsim+labels for {len(SCORES)-1} (layer,target) cells "
              f"to {args.dump_scores}]")


def cli():
    p = options.get_validation_parser()
    p.add_argument("--dude-root", type=str, required=True)
    p.add_argument("--test-root", type=str, default="",
                   help="external dataset root for held-out test (e.g. LIT-PCBA); "
                        "train/val stay on --dude-root. Empty = in-domain DUD-E test.")
    p.add_argument("--select-on-test-root", action="store_true",
                   help="fair operating-point check: W still trained on DUD-E, but "
                        "select layer+alpha on a LIT-PCBA val split and report on a "
                        "disjoint LIT-PCBA test split (interleaved).")
    p.add_argument("--dump-metrics", type=str, default="",
                   help="write per-(layer,target) metrics for every operating point to "
                        "this JSON, so LOO-CV / any split can be computed offline.")
    p.add_argument("--dump-scores", type=str, default="",
                   help="write per-mol dot+maxsim+labels per (layer,target) to this pickle, "
                        "so the top-K deployment-cascade sweep can be computed offline.")
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
    p.add_argument("--layer", type=int, default=15)        # single layer (used if --layers is empty)
    p.add_argument("--layers", type=str, default="15,14,13,12")  # comma list; sweep + select on val
    p.add_argument("--random-neg", action="store_true")   # ablation: disable hard-negative mining
    p.add_argument("--share-w", action="store_true")       # ablation: one shared projection
    p.add_argument("--with-raw", action="store_true")      # ablation: also report raw-MaxSim (no W)
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
