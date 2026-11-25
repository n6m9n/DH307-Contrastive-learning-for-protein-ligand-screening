"""
conformer_contact.py -- contact-recovery robustness to the ligand conformation
(advisor item 5). The Phase-2 contact test encoded the ligand in its CRYSTAL (bound)
pose. In real screening you do not have the bound pose, only a generated conformer. We
repeat the test with an RDKit ETKDG-generated conformer of the same ligand, keep the
CRYSTAL contacts as labels, and compare:
  * W on the crystal pose        (reference; = the Phase-2 result)
  * W on a generated conformer   (the screening-realistic input)
  * raw (no W) on the conformer  (ablation)
  * an element-based donor/acceptor + hydrophobic compatibility baseline (no learning)
Ground truth: ligand atom i contacts pocket atom j iff their CRYSTAL coords are < 4 A.
Atom order is shared between crystal and conformer (same RDKit heavy-atom indexing), so
crystal labels apply to the conformer embeddings directly.

Run (lab PC, env drugclip, from ~/drugclip/DrugCLIP):
  python conformer_contact.py --user-dir ./unimol ./data --valid-subset test \
    --task drugclip --loss in_batch_softmax --arch drugclip --seed 1 \
    --path ../checkpoint_best.pt --max-pocket-atoms 511 --batch-size 8 --num-workers 2 \
    --ddp-backend=c10d --dude-root ~/drugclip/dude_data/data/protein/DUD-E/raw/all \
    --litpcba-data ~/drugclip/litpcba_data --litpcba-raw ~/litpcba_raw \
    --layers 12,15 --contact-dist 4.0 --n-conformers 1 \
    --dump ~/conformer_contact.json 2>&1 | tee ~/conformer_contact.log ; touch ~/conformer_contact.DONE
"""
import os, sys, json, random, pickle, shutil, argparse
import numpy as np
import torch
import torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore

from rerank_eval import encode_target, train_W, _DS
from contact_recover import (read_pocket_coords, auroc, complex_metrics, boot,
                             boot_delta, element_of, pick_litpcba_template, write_lmdb, SCRATCH)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# ---------- ligand: crystal heavy coords + a generated conformer, shared atom order ----------
def read_pocket_elems(pocket_lmdb):
    import lmdb
    env = lmdb.open(pocket_lmdb, subdir=False, readonly=True, lock=False,
                    readahead=False, meminit=False)
    with env.begin() as txn:
        d = pickle.loads(txn.get(b"0"))
    env.close()
    return [element_of(n) for n in d["pocket_atoms"] if element_of(n) != "H"]


def ligand_crystal_and_conformer(mol2_path, seed):
    """Return (elems, crystal_coords[N,3], conformer_coords[N,3]) for heavy atoms in a
    shared order, or None on failure. Crystal coords come from the mol2; the conformer is
    a fresh ETKDG embedding of the same graph (MMFF-optimized)."""
    mol = Chem.MolFromMol2File(mol2_path, removeHs=False, sanitize=True)
    if mol is None:   # RDKit mol2 parser is fragile; retry without sanitize, then sanitize loosely
        mol = Chem.MolFromMol2File(mol2_path, removeHs=False, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                             ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                             ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
        except Exception:
            return None
    try:
        heavy = Chem.RemoveHs(mol)
    except Exception:
        try:
            heavy = Chem.RemoveHs(mol, sanitize=False)
        except Exception:
            return None
    if heavy.GetNumConformers() == 0:
        return None
    elems = [a.GetSymbol() for a in heavy.GetAtoms()]
    crystal = heavy.GetConformer().GetPositions().astype(float)  # heavy, crystal order
    # generate a conformer on the same heavy-atom graph (add H for embedding, then drop)
    try:
        molH = Chem.AddHs(heavy)   # reuse `heavy` (already sanitized); adds H without coords
        ok = AllChem.EmbedMolecule(molH, AllChem.ETKDGv3())
        if ok != 0:
            ok = AllChem.EmbedMolecule(molH, useRandomCoords=True, randomSeed=seed)
            if ok != 0:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(molH)
        except Exception:
            pass
        conf_noH = Chem.RemoveHs(molH, sanitize=False)
    except Exception:
        return None
    if conf_noH.GetNumAtoms() != len(elems) or conf_noH.GetNumConformers() == 0:
        return None
    conformer = conf_noH.GetConformer().GetPositions().astype(float)
    return elems, crystal, conformer


# ---------- element-based donor/acceptor + hydrophobic compatibility prior ----------
HBOND = {"N", "O"}
HYDRO = {"C", "S", "Cl", "Br", "F", "I"}

def chem_compat_matrix(lig_elems, poc_elems):
    """Coarse non-learned prior: H-bond-capable pairs (N/O with N/O) and hydrophobic pairs
    score high; mixed pairs low. Element-only proxy for donor/acceptor + hydrophobic
    compatibility (no explicit donor/acceptor assignment)."""
    le = np.array([1 if e in HBOND else (2 if e in HYDRO else 0) for e in lig_elems])
    pe = np.array([1 if e in HBOND else (2 if e in HYDRO else 0) for e in poc_elems])
    C = np.full((len(le), len(pe)), 0.2)
    C[(le[:, None] == 1) & (pe[None, :] == 1)] = 1.0   # H-bond compatible
    C[(le[:, None] == 2) & (pe[None, :] == 2)] = 1.0   # hydrophobic compatible
    return C


def build_complexes(args):
    comps = []
    dude_targets = sorted(os.listdir(args.dude_root))
    for t in dude_targets[:args.n_test]:
        poc = os.path.join(args.dude_root, t, "pocket.lmdb")
        lig = os.path.join(args.dude_root, t, "crystal_ligand.mol2")
        if os.path.exists(poc) and os.path.exists(lig):
            comps.append((f"dude_{t}", poc, lig))
    if args.litpcba_data and args.litpcba_raw:
        for t in sorted(os.listdir(args.litpcba_data)):
            poc = os.path.join(args.litpcba_data, t, "pocket.lmdb")
            raw_tdir = os.path.join(args.litpcba_raw, t)
            if os.path.exists(poc) and os.path.isdir(raw_tdir):
                lig, _ = pick_litpcba_template(raw_tdir)
                if os.path.exists(lig):
                    comps.append((f"ltp_{t}", poc, lig))
    return comps


def stage_conformer(name, elems, conformer_coords):
    d = os.path.join(SCRATCH, name)
    os.makedirs(d, exist_ok=True)
    poc_src = os.path.join(d, "pocket.lmdb")
    entry = {"atoms": list(elems), "coordinates": [np.asarray(conformer_coords, float)],
             "smi": name, "label": 1}
    write_lmdb([entry], os.path.join(d, "mols.lmdb"))
    return d


def main(args):
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
    train_ts = dude_targets[args.n_test + args.n_val: args.n_test + args.n_val + args.n_train]

    os.makedirs(SCRATCH, exist_ok=True)
    comps = build_complexes(args)
    # prepare ligand crystal+conformer, pocket coords/elems, stage conformer lmdbs
    staged = []
    for name, poc, lig in comps:
        try:
            res = ligand_crystal_and_conformer(lig, args.seed)
        except Exception as e:
            print(f"  [skip] {name}: RDKit error {type(e).__name__}"); continue
        if res is None:
            print(f"  [skip] {name}: RDKit could not parse/embed ligand"); continue
        elems, crystal, conformer = res
        pcoord = read_pocket_coords(poc)
        pelems = read_pocket_elems(poc)
        d = stage_conformer(name, elems, conformer)
        # symlink the real pocket so the loader encodes the crystal pocket
        pdst = os.path.join(d, "pocket.lmdb")
        if os.path.islink(pdst) or os.path.exists(pdst):
            os.remove(pdst)
        os.symlink(os.path.abspath(poc), pdst)
        rmsd = float(np.sqrt(((crystal - crystal.mean(0)) ** 2).sum(1).mean()))  # placeholder scale
        staged.append((name, elems, crystal, conformer, pcoord, pelems))
    print(f"\nstaged {len(staged)} complexes "
          f"({sum(n.startswith('dude_') for n,*_ in staged)} DUD-E + "
          f"{sum(n.startswith('ltp_') for n,*_ in staged)} LIT-PCBA)")
    layers = [int(x) for x in args.layers.split(",")]

    dump = {}
    for layer in layers:
        Wm, Wp = train_W(args, task, model, dev, use_cuda, full, layer, train_ts)
        Wm.eval(); Wp.eval()
        rows = []
        for name, elems, crystal, conformer, pcoord, pelems in staged:
            for k in [k for k in _DS if k[1] == name]:
                del _DS[k]
            p_atoms, _pcls, all_atoms, _labs, _cls = encode_target(
                task, model, SCRATCH, name, use_cuda, None, layer, full)
            if len(all_atoms) == 0:
                print(f"  [skip] {name}: encoder returned no ligand"); continue
            lig = all_atoms[0].to(dev).float()
            poc = p_atoms.float()
            if lig.shape[0] != len(elems) or poc.shape[0] != pcoord.shape[0]:
                print(f"  [skip] {name}: token/coord mismatch lig {lig.shape[0]}/{len(elems)} "
                      f"poc {poc.shape[0]}/{pcoord.shape[0]}"); continue
            with torch.no_grad():
                Lw = F.normalize(Wm(lig), dim=-1); Pw = F.normalize(Wp(poc), dim=-1)
                Cw = (Lw @ Pw.T).cpu().numpy()                    # W on conformer
                Lr = F.normalize(lig, dim=-1); Pr = F.normalize(poc, dim=-1)
                Cr = (Lr @ Pr.T).cpu().numpy()                    # raw on conformer
            D = np.linalg.norm(crystal[:, None, :] - pcoord[None, :, :], axis=2)
            G = (D < args.contact_dist).astype(float)             # CRYSTAL contacts
            Cchem = chem_compat_matrix(elems, pelems)             # non-learned prior
            mW = complex_metrics(Cw, G); mR = complex_metrics(Cr, G); mC = complex_metrics(Cchem, G)
            if mW is None or mR is None or mC is None:
                print(f"  [skip] {name}: no contacting atoms"); continue
            rows.append((name, mW[0], mR[0], mC[0], mW[1], mR[1], mC[1], mW[2]))
            torch.cuda.empty_cache()

        names = [r[0] for r in rows]
        aucW = [r[1] for r in rows]; aucR = [r[2] for r in rows]; aucC = [r[3] for r in rows]
        t1W = [r[4] for r in rows]; t1R = [r[5] for r in rows]; t1C = [r[6] for r in rows]; t1rand = [r[7] for r in rows]
        dump[str(layer)] = {r[0]: {"aucW_conf": r[1], "aucRaw_conf": r[2], "aucChem": r[3],
                                   "t1W_conf": r[4], "t1Raw_conf": r[5], "t1Chem": r[6],
                                   "t1rand": r[7]} for r in rows}
        print(f"\n===== layer {layer}: contact recovery with GENERATED conformers, "
              f"{len(rows)} complexes (crystal contacts as labels) =====")
        for tag, v in [("AUROC W(conformer)", aucW), ("AUROC raw(conformer)", aucR),
                       ("AUROC chem-baseline", aucC),
                       ("top1  W(conformer)", t1W), ("top1  chem", t1C), ("top1  random", t1rand)]:
            m, lo, hi = boot(v)
            print(f"  {tag:22s} mean {m:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
        for a, b, lab in [(aucW, aucR, "W - raw   (conformer)"),
                          (aucW, aucC, "W - chem  (conformer)")]:
            m, lo, hi, s = boot_delta(a, b)
            print(f"  AUROC {lab:22s} Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] {s}")
        for pfx, labn in [("dude_", "DUD-E"), ("ltp_", "LIT-PCBA")]:
            idx = [i for i, n in enumerate(names) if n.startswith(pfx)]
            if idx:
                print(f"    [{labn:8s} n={len(idx)}] AUROC W(conf) "
                      f"{np.mean([aucW[i] for i in idx]):.4f}  chem {np.mean([aucC[i] for i in idx]):.4f}")
        del Wm, Wp
        torch.cuda.empty_cache()

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump(dump, f)
        print(f"\n[dumped per-complex conformer-contact metrics to {args.dump}]")
    shutil.rmtree(SCRATCH, ignore_errors=True)


def cli():
    p = options.get_validation_parser()
    p.add_argument("--dude-root", type=str, required=True)
    p.add_argument("--litpcba-data", type=str, default="")
    p.add_argument("--litpcba-raw", type=str, default="")
    p.add_argument("--contact-dist", type=float, default=4.0)
    p.add_argument("--n-conformers", type=int, default=1)
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
    p.add_argument("--layers", type=str, default="12,15")
    p.add_argument("--random-neg", action="store_true")
    p.add_argument("--share-w", action="store_true")
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
