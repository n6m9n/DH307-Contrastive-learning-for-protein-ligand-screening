"""
contact_recover.py -- Phase 2 contact-recovery test for the DrugCLIP late-
interaction reranker.  Does the LEARNED atom-atom correspondence actually land
on PHYSICAL contacts?  Geometric ground truth (no PLIP/OpenBabel needed): we
have crystal protein + crystal bound-ligand coordinates for every holo template,
so a ligand atom "contacts" a pocket atom if they are within `--contact-dist` A.

For each holo complex (crystal pocket + its own crystal bound ligand):
  * encode pocket and ligand per-atom tokens at layer L (frozen DrugCLIP encoders)
  * project through the DUD-E-trained W, L2-normalize  -> cosine matrix C (Nl x Np)
  * ground truth G_ij = 1 iff ||lig_i - poc_j|| < contact-dist  (crystal frame)
  * for each CONTACTING ligand atom i: rank pocket atoms by C[i]; measure
      - AUROC (do true contacts out-rank non-contacts)
      - top-1 hit (is argmax_j C[i,j] a true contact)
  * baselines: raw cosine WITHOUT W (the key ablation); random (chance).
Aggregated per complex, bootstrapped over complexes.  W is trained on DUD-E only
(targets[n_test+n_val:...]); the evaluated complexes (20 held-out DUD-E test +
15 LIT-PCBA) are never in W-training, so there is no leakage.

Reuses rerank_eval's model/encoder/W machinery unchanged.

Run (on lab PC, env drugclip, from ~/drugclip/DrugCLIP):
  python contact_recover.py --user-dir ./unimol ./data --valid-subset test \
    --task drugclip --loss in_batch_softmax --arch drugclip --seed 1 \
    --path ../checkpoint_best.pt --max-pocket-atoms 511 --batch-size 8 \
    --num-workers 2 --ddp-backend=c10d \
    --dude-root ~/drugclip/dude_data/data/protein/DUD-E/raw/all \
    --litpcba-data ~/drugclip/litpcba_data --litpcba-raw ~/litpcba_raw \
    --layers 12,15 --contact-dist 4.0 \
    --dump ~/contact_permetrics.json 2>&1 | tee ~/contact.log ; touch ~/contact.DONE
"""
import os, sys, json, random, pickle, shutil, argparse
import numpy as np
import torch
import torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore

# reuse the exact encoder / W-training / dataset machinery from the main experiment
from rerank_eval import (enc, atom_mask_from_pad, encode_target, train_W,
                         maxsim, _DS)

SCRATCH = os.path.expanduser("~/.cr_scratch")   # temp one-ligand datasets live here
PLIP_GT = None                                   # {name: [{type, lig, prot}, ...]} if --plip-gt


# ---------- minimal mol2 heavy-atom reader (same convention as build_litpcba) ----------
def element_of(atom_name):
    a = atom_name.strip()
    return a[1] if (a and a[0].isdigit()) else a[0]


def read_mol2_heavy(path):
    """Return (element list, (n,3) coords) for heavy atoms of the TRIPOS ATOM block."""
    names, coords = [], []
    in_atom = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("@<TRIPOS>"):
                in_atom = s.startswith("@<TRIPOS>ATOM")
                continue
            if not in_atom or not s:
                continue
            p = s.split()
            if len(p) < 5:
                continue
            names.append(p[1])
            coords.append([float(p[2]), float(p[3]), float(p[4])])
    coords = np.asarray(coords, dtype=float)
    keep = [i for i, n in enumerate(names) if element_of(n) != "H"]
    elems = [element_of(names[i]) for i in keep]
    return elems, coords[keep]


def write_lmdb(data, path):
    import lmdb
    if os.path.exists(path):
        os.remove(path)
    env = lmdb.open(path, subdir=False, readonly=False, lock=False,
                    readahead=False, meminit=False, map_size=1 << 34)
    with env.begin(write=True) as txn:
        for i, d in enumerate(data):
            txn.put(str(i).encode("ascii"), pickle.dumps(d))
    env.sync(); env.close()


def read_pocket_coords(pocket_lmdb):
    """Pocket heavy-atom coords in the SAME order the encoder sees them. The loader
    runs RemoveHydrogenPocketDataset, so we drop H here too (by atom name); no
    cropping happens (pockets << max_pocket_atoms). Order is otherwise preserved."""
    import lmdb
    env = lmdb.open(pocket_lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        d = pickle.loads(txn.get(b"0"))
    env.close()
    names = d["pocket_atoms"]
    coords = np.asarray(d["pocket_coordinates"], dtype=float)
    keep = [i for i, n in enumerate(names) if element_of(n) != "H"]
    return coords[keep]


def pick_litpcba_template(raw_tdir):
    prots = sorted(f for f in os.listdir(raw_tdir) if f.endswith("_protein.mol2"))
    pdbid = prots[0].split("_")[0]          # alphabetical-first == what build_litpcba used
    return os.path.join(raw_tdir, f"{pdbid}_ligand.mol2"), pdbid


# ---------- metrics ----------
def auroc(scores, labels):
    """P(score[pos] > score[neg]) via rank sum. labels in {0,1}."""
    pos = int(labels.sum()); neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return (ranks[labels == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg)


def complex_metrics(C, G):
    """C, G: (Nl, Np). Return per-complex means over contacting ligand atoms."""
    aW, t1, trand = [], [], []
    for i in range(C.shape[0]):
        g = G[i]
        if g.sum() == 0 or g.sum() == len(g):
            continue
        a = auroc(C[i], g)
        if a is None:
            continue
        aW.append(a)
        t1.append(float(g[np.argmax(C[i])]))
        trand.append(float(g.mean()))      # expected top-1 of a random pick
    if not aW:
        return None
    return np.mean(aW), np.mean(t1), np.mean(trand), len(aW)


def build_typed_G(inters, lcoord, pcoord, lig_tol, prot_tol):
    """Build the ATOM-PAIR typed ground truth from PLIP interactions.

    PLIP's ligcoo/protcoo are the genuine interacting ATOMS only for atom-level types
    (hydrophobic, H-bond, halogen); for group-level types (pi-stacking, pi-cation,
    salt bridges) they are ring/charge CENTROIDS, and water bridges are water-mediated.
    We therefore map only level=="atom" interactions to token pairs (their coords
    coincide with real atoms, so nearest-token mapping is exact), and count the rest
    separately. Returns (G, n_mapped_atom, n_atom, level_counts, max_residual_A)."""
    Nl, Np = len(lcoord), len(pcoord)
    G = np.zeros((Nl, Np), dtype=float)
    mapped = 0; n_atom = 0; max_res = 0.0
    levels = {}
    for it in inters:
        lev = it.get("level", "other")
        levels[lev] = levels.get(lev, 0) + 1
        if lev != "atom":
            continue
        n_atom += 1
        lc = np.asarray(it["lig"], dtype=float); pc = np.asarray(it["prot"], dtype=float)
        di = np.linalg.norm(lcoord - lc[None, :], axis=1); i = int(di.argmin())
        dj = np.linalg.norm(pcoord - pc[None, :], axis=1); j = int(dj.argmin())
        if di[i] <= lig_tol and dj[j] <= prot_tol:
            G[i, j] = 1.0; mapped += 1
            max_res = max(max_res, float(di[i]), float(dj[j]))
    return G, mapped, n_atom, levels, max_res


def boot(vals, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    m = np.array([rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return vals.mean(), lo, hi


def boot_delta(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    m = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    sig = "*" if (lo > 0 or hi < 0) else ""
    return d.mean(), lo, hi, sig


# ---------- build the list of holo complexes ----------
def build_complexes(args):
    """Return list of (name, pocket_lmdb, ligand_mol2). Also stage a one-ligand
    mols.lmdb + symlinked pocket under SCRATCH/<name>/ for the encoder."""
    comps = []
    os.makedirs(SCRATCH, exist_ok=True)

    dude_targets = sorted(os.listdir(args.dude_root))
    dude_test = dude_targets[:args.n_test]        # the 20 held-out test targets
    for t in dude_test:
        poc = os.path.join(args.dude_root, t, "pocket.lmdb")
        lig = os.path.join(args.dude_root, t, "crystal_ligand.mol2")
        if os.path.exists(poc) and os.path.exists(lig):
            comps.append((f"dude_{t}", poc, lig))

    if args.litpcba_data and args.litpcba_raw:
        for t in sorted(os.listdir(args.litpcba_data)):
            poc = os.path.join(args.litpcba_data, t, "pocket.lmdb")
            raw_tdir = os.path.join(args.litpcba_raw, t)
            if not (os.path.exists(poc) and os.path.isdir(raw_tdir)):
                continue
            lig, _ = pick_litpcba_template(raw_tdir)
            if os.path.exists(lig):
                comps.append((f"ltp_{t}", poc, lig))

    staged = []
    for name, poc, lig in comps:
        elems, lcoord = read_mol2_heavy(lig)
        if len(elems) == 0:
            print(f"  [skip] {name}: empty ligand")
            continue
        d = os.path.join(SCRATCH, name)
        os.makedirs(d, exist_ok=True)
        # symlink the prebuilt pocket; write a one-entry mols.lmdb (crystal ligand)
        pdst = os.path.join(d, "pocket.lmdb")
        if os.path.islink(pdst) or os.path.exists(pdst):
            os.remove(pdst)
        os.symlink(os.path.abspath(poc), pdst)
        entry = {"atoms": elems, "coordinates": [lcoord],
                 "smi": name, "label": 1}
        write_lmdb([entry], os.path.join(d, "mols.lmdb"))
        pcoord = read_pocket_coords(poc)
        staged.append((name, lcoord, pcoord, len(elems)))
    return staged


def main(args):
    global PLIP_GT
    if args.plip_gt:
        with open(args.plip_gt) as f:
            PLIP_GT = json.load(f)
        print(f"[typed] loaded PLIP ground truth for {len(PLIP_GT)} complexes from "
              f"{args.plip_gt} (lig_tol {args.plip_lig_tol} A, prot_tol {args.plip_prot_tol} A)")
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

    dude_targets = sorted(os.listdir(args.dude_root))
    train_ts = dude_targets[args.n_test + args.n_val:
                            args.n_test + args.n_val + args.n_train]

    staged = build_complexes(args)
    print(f"\nstaged {len(staged)} holo complexes "
          f"({sum(n.startswith('dude_') for n,_,_,_ in staged)} DUD-E + "
          f"{sum(n.startswith('ltp_') for n,_,_,_ in staged)} LIT-PCBA)")
    layers = [int(x) for x in args.layers.split(",")]
    print(f"layers: {layers}   contact-dist: {args.contact_dist} A   "
          f"W trained on {len(train_ts)} DUD-E targets")

    dump = {}
    for layer in layers:
        Wm, Wp = train_W(args, task, model, dev, use_cuda, full, layer, train_ts)
        Wm.eval(); Wp.eval()
        rows = []   # (name, aucW, aucRaw, t1W, t1Raw, t1Rand, n_atoms)
        rows_typed = []   # (name, aucW, aucRaw, t1W, t1Raw, t1Rand, n_atoms, nmap, natom)
        LEVEL_COUNTS = {}   # total interactions by level (atom/group/water/metal)
        RESIDUALS = []      # per-complex max atom-pair mapping residual (A)
        for name, lcoord, pcoord, n_elems in staged:
            # clear any cached dataset for this scratch target so we re-read fresh
            for k in [k for k in _DS if k[1] == name]:
                del _DS[k]
            p_atoms, _pcls, all_atoms, _labs, _cls = encode_target(
                task, model, SCRATCH, name, use_cuda, None, layer, full)
            if len(all_atoms) == 0:
                print(f"  [skip] {name}: encoder returned no ligand"); continue
            lig = all_atoms[0].to(dev).float()          # (Nl, 512)
            poc = p_atoms.float()                        # (Np, 512)
            # alignment guard: tokens must match stored coordinate counts
            if lig.shape[0] != lcoord.shape[0] or poc.shape[0] != pcoord.shape[0]:
                print(f"  [skip] {name}: token/coord mismatch "
                      f"lig {lig.shape[0]}/{lcoord.shape[0]} poc {poc.shape[0]}/{pcoord.shape[0]}")
                continue
            with torch.no_grad():
                Lw = F.normalize(Wm(lig), dim=-1); Pw = F.normalize(Wp(poc), dim=-1)
                Cw = (Lw @ Pw.T).cpu().numpy()
                Lr = F.normalize(lig, dim=-1); Pr = F.normalize(poc, dim=-1)
                Cr = (Lr @ Pr.T).cpu().numpy()
            # ---- typed (PLIP) ground truth, if provided (secondary analysis) ----
            if PLIP_GT is not None:
                inters = PLIP_GT.get(name, [])
                Gt, nmap, natom, levels, maxres = build_typed_G(
                    inters, lcoord, pcoord, args.plip_lig_tol, args.plip_prot_tol)
                for k, v in levels.items():
                    LEVEL_COUNTS[k] = LEVEL_COUNTS.get(k, 0) + v
                RESIDUALS.append(maxres)
                mWt = complex_metrics(Cw, Gt)
                mRt = complex_metrics(Cr, Gt)
                if mWt is not None and mRt is not None:
                    rows_typed.append((name, mWt[0], mRt[0], mWt[1], mRt[1],
                                       mWt[2], mWt[3], nmap, natom))
                else:
                    print(f"  [typed-skip] {name}: {nmap}/{natom} atom-level mapped, "
                          f"no scorable ligand atom")
            D = np.linalg.norm(lcoord[:, None, :] - pcoord[None, :, :], axis=2)
            G = (D < args.contact_dist).astype(float)
            mW = complex_metrics(Cw, G)
            mR = complex_metrics(Cr, G)
            if mW is None or mR is None:
                print(f"  [skip] {name}: no contacting ligand atoms"); continue
            rows.append((name, mW[0], mR[0], mW[1], mR[1], mW[2], mW[3]))
            torch.cuda.empty_cache()

        names = [r[0] for r in rows]
        aucW = [r[1] for r in rows]; aucR = [r[2] for r in rows]
        t1W = [r[3] for r in rows]; t1R = [r[4] for r in rows]; t1Rand = [r[5] for r in rows]
        dump[str(layer)] = {r[0]: {"aucW": r[1], "aucRaw": r[2], "t1W": r[3],
                                   "t1Raw": r[4], "t1Rand": r[5], "natoms": r[6]}
                            for r in rows}

        print(f"\n===== layer {layer}: contact recovery over {len(rows)} complexes =====")
        for tag, v in [("AUROC  W", aucW), ("AUROC  raw", aucR),
                       ("top1   W", t1W), ("top1   raw", t1R), ("top1   random", t1Rand)]:
            m, lo, hi = boot(v)
            print(f"  {tag:14s}  mean {m:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
        dm, lo, hi, s = boot_delta(aucW, aucR)
        print(f"  AUROC  W - raw   Δ {dm:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}] {s}")
        dm, lo, hi, s = boot_delta(t1W, t1R)
        print(f"  top1   W - raw   Δ {dm:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}] {s}")
        dm, lo, hi, s = boot_delta(t1W, t1Rand)
        print(f"  top1   W - rand  Δ {dm:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}] {s}")
        # split DUD-E vs LIT-PCBA
        for pfx, lab in [("dude_", "DUD-E"), ("ltp_", "LIT-PCBA")]:
            idx = [i for i, n in enumerate(names) if n.startswith(pfx)]
            if idx:
                print(f"    [{lab:8s} n={len(idx)}] AUROC W {np.mean([aucW[i] for i in idx]):.4f}  "
                      f"raw {np.mean([aucR[i] for i in idx]):.4f}  | "
                      f"top1 W {np.mean([t1W[i] for i in idx]):.4f}  "
                      f"raw {np.mean([t1R[i] for i in idx]):.4f}  "
                      f"rand {np.mean([t1Rand[i] for i in idx]):.4f}")

        # ---- typed (PLIP) results block ----
        if PLIP_GT is not None and rows_typed:
            tn = [r[0] for r in rows_typed]
            taW = [r[1] for r in rows_typed]; taR = [r[2] for r in rows_typed]
            tt1W = [r[3] for r in rows_typed]; tt1R = [r[4] for r in rows_typed]
            tt1Rand = [r[5] for r in rows_typed]
            nmap_tot = sum(r[7] for r in rows_typed); natom_tot = sum(r[8] for r in rows_typed)
            dump[f"{layer}|typed"] = {r[0]: {"aucW": r[1], "aucRaw": r[2], "t1W": r[3],
                                             "t1Raw": r[4], "t1Rand": r[5], "natoms": r[6],
                                             "nmap": r[7], "natom": r[8]}
                                      for r in rows_typed}
            dump[f"{layer}|typed_meta"] = {"level_counts": LEVEL_COUNTS,
                                           "max_residual_A": max(RESIDUALS) if RESIDUALS else 0.0}
            na = LEVEL_COUNTS.get("atom", 0)
            excl = ", ".join(f"{k} {v}" for k, v in sorted(LEVEL_COUNTS.items()) if k != "atom")
            print(f"\n----- layer {layer}: ATOM-LEVEL TYPED (PLIP) recovery over {len(rows_typed)} "
                  f"complexes -----")
            print(f"  atom-level interactions (hydrophobic+H-bond+halogen): {na}, "
                  f"of which {nmap_tot} mapped to token pairs "
                  f"(max residual {max(RESIDUALS) if RESIDUALS else 0.0:.2f} A)")
            print(f"  excluded (reported separately): {excl or 'none'}")
            for tag, v in [("AUROC  W", taW), ("AUROC  raw", taR),
                           ("top1   W", tt1W), ("top1   raw", tt1R), ("top1   random", tt1Rand)]:
                m, lo, hi = boot(v)
                print(f"  {tag:14s}  mean {m:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
            dm, lo, hi, s = boot_delta(taW, taR)
            print(f"  AUROC  W - raw   Δ {dm:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}] {s}")
            dm, lo, hi, s = boot_delta(tt1W, tt1R)
            print(f"  top1   W - raw   Δ {dm:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}] {s}")
            dm, lo, hi, s = boot_delta(tt1W, tt1Rand)
            print(f"  top1   W - rand  Δ {dm:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}] {s}")
            for pfx, lab in [("dude_", "DUD-E"), ("ltp_", "LIT-PCBA")]:
                idx = [i for i, n in enumerate(tn) if n.startswith(pfx)]
                if idx:
                    print(f"    [{lab:8s} n={len(idx)}] AUROC W {np.mean([taW[i] for i in idx]):.4f}  "
                          f"raw {np.mean([taR[i] for i in idx]):.4f}  | "
                          f"top1 W {np.mean([tt1W[i] for i in idx]):.4f}  "
                          f"raw {np.mean([tt1R[i] for i in idx]):.4f}  "
                          f"rand {np.mean([tt1Rand[i] for i in idx]):.4f}")

        del Wm, Wp
        torch.cuda.empty_cache()

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dump, f)
        print(f"\n[dumped per-complex contact metrics to {args.dump}]")
    # cleanup scratch
    shutil.rmtree(SCRATCH, ignore_errors=True)


def cli():
    p = options.get_validation_parser()
    p.add_argument("--dude-root", type=str, required=True)
    p.add_argument("--litpcba-data", type=str, default="",
                   help="built LIT-PCBA root (per-target pocket.lmdb)")
    p.add_argument("--litpcba-raw", type=str, default="",
                   help="raw LIT-PCBA root (per-target <pdbid>_ligand.mol2 templates)")
    p.add_argument("--contact-dist", type=float, default=4.0)
    p.add_argument("--plip-gt", type=str, default="",
                   help="JSON from plip_gt.py; adds a TYPED-interaction recovery block")
    p.add_argument("--plip-lig-tol", type=float, default=2.0,
                   help="max A from a PLIP ligand coord to a ligand token to map it")
    p.add_argument("--plip-prot-tol", type=float, default=2.5,
                   help="max A from a PLIP protein coord to a pocket token (centroids need slack)")
    p.add_argument("--dump", type=str, default="")
    # W-training hyperparameters (same defaults as rerank_eval)
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
    p.add_argument("--layers", type=str, default="12,15")
    p.add_argument("--random-neg", action="store_true")
    p.add_argument("--share-w", action="store_true")
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
