"""
build_litpcba.py  --  turn raw LIT-PCBA into DrugCLIP mols.lmdb + pocket.lmdb,
using the SAME schema as py_scripts/write_dude_multi.py so rerank_eval.py /
cost_recall.py run on it unchanged.

Raw LIT-PCBA layout (from full_data.tgz):
    <raw-root>/<TARGET>/actives.smi            "<SMILES> <CID>" per line
    <raw-root>/<TARGET>/inactives.smi
    <raw-root>/<TARGET>/<pdbid>_protein.mol2   (one or more templates)
    <raw-root>/<TARGET>/<pdbid>_ligand.mol2

Output (mirrors DUD-E):
    <out-root>/<TARGET>/mols.lmdb     entries {'atoms','coordinates','smi','label'}
                                      actives (label 1) first, then inactives (0)
    <out-root>/<TARGET>/pocket.lmdb   one entry
                                      {'pocket','pocket_index','pocket_atoms','pocket_coordinates'}

Schema faithfulness vs write_dude_multi.py:
  * conformer: AddHs -> EmbedMultipleConfs(numConfs=1) -> MMFFOptimize -> RemoveHs
  * mols entry: atoms=element symbols, coordinates=list of (n,3) arrays, smi, label
  * pocket: residues with ANY atom within `raid` A of ANY reference-ligand atom,
            store that residue's atom_name strings (first char = element, as the
            model's AffinityPocketDataset expects) + per-atom coordinates
  * write_lmdb: subdir=False, keys str(num).encode('ascii'), pickle.dumps

LIT-PCBA adaptations:
  * protein read from .mol2 (PandasMol2), not .pdb  -> residue id = subst_name
  * LIT-PCBA mol2 carry explicit H; DUD-E PDBs did not. We drop H from both
    protein and reference ligand BEFORE the 6 A selection so the pocket matches
    DUD-E's heavy-atom definition (the loader drops pocket H anyway).
  * multi-template targets: default to the alphabetically-first *_protein.mol2;
    override per target with --template "TP53:3zme,ESR1_ant:2b1z".
"""
import argparse
import multiprocessing as mp
import os
import pickle
import random

import lmdb
import numpy as np
import tqdm
from rdkit import Chem, RDLogger
import rdkit.Chem.AllChem as AllChem

RDLogger.DisableLog("rdApp.*")

PILOT = ["TP53", "ESR1_ant", "PPARG"]


def element_of(atom_name):
    # mirrors AffinityPocketDataset.pocket_atom: first alpha char is the element
    a = atom_name.strip()
    return a[1] if (a and a[0].isdigit()) else a[0]


# ---------- conformer generation (identical flags to write_dude_multi) ----------
def gen_conformation(mol, num_conf=1, num_worker=5):
    try:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(
            mol, numConfs=num_conf, numThreads=num_worker,
            pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False,
        )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker)
        except Exception:
            pass
        mol = Chem.RemoveHs(mol)
    except Exception:
        return None
    if mol.GetNumConformers() == 0:
        return None
    return mol


def smi_to_entry(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = gen_conformation(mol)
    if mol is None:
        return None
    coords = [np.array(mol.GetConformer(i).GetPositions())
              for i in range(mol.GetNumConformers())]
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    return {"atoms": atoms, "coordinates": coords, "smi": Chem.MolToSmiles(mol)}


def read_smis(path, cap, seed):
    smis = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            smis.append(line.split()[0])
    if cap and cap > 0 and len(smis) > cap:
        rng = random.Random(seed)
        smis = rng.sample(smis, cap)
    return smis


# ---------- pocket from mol2 ----------
def read_mol2_atoms(path, drop_h=True):
    """Minimal TRIPOS mol2 ATOM parser (no biopandas).
    Columns: atom_id atom_name x y z atom_type [subst_id subst_name [charge ...]]
    Returns element-carrying atom names, (n,3) coords, residue ids (subst_name)."""
    names, coords, subst = [], [], []
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
            subst.append(p[7] if len(p) > 7 else "LIG0")
    coords = np.asarray(coords, dtype=float)
    if drop_h and len(names):
        keep = [i for i, n in enumerate(names) if element_of(n) != "H"]
        names = [names[i] for i in keep]
        coords = coords[keep]
        subst = [subst[i] for i in keep]
    return names, coords, subst


def build_pocket(protein_path, ligand_path, pocket_index, raid, target):
    p_names, p_coord, p_res = read_mol2_atoms(protein_path, drop_h=True)
    _, l_coord, _ = read_mol2_atoms(ligand_path, drop_h=True)
    # residue is in the pocket if ANY of its atoms is within raid of ANY ligand atom
    d = np.linalg.norm(p_coord[:, None, :] - l_coord[None, :, :], axis=2)  # (P, L)
    within = (d < raid).any(axis=1)
    pocket_res = {p_res[i] for i in range(len(p_res)) if within[i]}
    idx = [i for i, r in enumerate(p_res) if r in pocket_res]
    pocket_atoms = [p_names[i] for i in idx]
    pocket_coord = [p_coord[i] for i in idx]
    return {
        "pocket": target,
        "pocket_index": pocket_index,
        "pocket_atoms": pocket_atoms,
        "pocket_coordinates": pocket_coord,
    }


def write_lmdb(data, path):
    if os.path.exists(path):
        os.remove(path)
    env = lmdb.open(path, subdir=False, readonly=False, lock=False,
                    readahead=False, meminit=False, map_size=1099511627776)
    with env.begin(write=True) as txn:
        for num, d in enumerate(data):
            txn.put(str(num).encode("ascii"), pickle.dumps(d))
    env.sync()
    env.close()


def pick_template(tdir, override):
    prots = sorted(f for f in os.listdir(tdir) if f.endswith("_protein.mol2"))
    if not prots:
        raise FileNotFoundError(f"no *_protein.mol2 in {tdir}")
    if override:
        for f in prots:
            if f.split("_")[0] == override:
                prots = [f]
                break
        else:
            raise ValueError(f"template {override} not found in {tdir} "
                             f"(have {[p.split('_')[0] for p in prots]})")
    pdbid = prots[0].split("_")[0]
    return (os.path.join(tdir, f"{pdbid}_protein.mol2"),
            os.path.join(tdir, f"{pdbid}_ligand.mol2"), pdbid)


def build_target(target, raw_root, out_root, idx, raid, cap_a, cap_i, nproc,
                 seed, override):
    tdir = os.path.join(raw_root, target)
    prot, lig, pdbid = pick_template(tdir, override)
    outdir = os.path.join(out_root, target)
    os.makedirs(outdir, exist_ok=True)

    poc = build_pocket(prot, lig, idx, raid, target)
    write_lmdb([poc], os.path.join(outdir, "pocket.lmdb"))

    act_smis = read_smis(os.path.join(tdir, "actives.smi"), cap_a, seed)
    ina_smis = read_smis(os.path.join(tdir, "inactives.smi"), cap_i, seed)

    data = []
    for smis, label, tag in [(act_smis, 1, "actives"), (ina_smis, 0, "inactives")]:
        ok = 0
        with mp.Pool(nproc) as pool:
            it = pool.imap_unordered(smi_to_entry, smis, chunksize=8)
            for e in tqdm.tqdm(it, total=len(smis), desc=f"{target}:{tag}"):
                if e is not None:
                    e["label"] = label
                    data.append(e)
                    ok += 1
        print(f"  {target} {tag}: {ok}/{len(smis)} conformers "
              f"({len(smis) - ok} failed)")
    write_lmdb(data, os.path.join(outdir, "mols.lmdb"))
    print(f"  {target}: template {pdbid}, pocket atoms {len(poc['pocket_atoms'])}, "
          f"mols {len(data)} -> {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True, help="extracted LIT-PCBA dir")
    ap.add_argument("--out-root", required=True, help="where to write per-target lmdbs")
    ap.add_argument("--targets", default=",".join(PILOT),
                    help="comma list, or 'all'")
    ap.add_argument("--max-actives", type=int, default=0, help="0 = all")
    ap.add_argument("--max-inactives", type=int, default=0, help="0 = all")
    ap.add_argument("--raid", type=float, default=6.0)
    ap.add_argument("--nproc", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--template", default="",
                    help='per-target overrides, e.g. "TP53:3zme,ESR1_ant:2b1z"')
    args = ap.parse_args()

    if args.targets.strip().lower() == "all":
        targets = sorted(d for d in os.listdir(args.raw_root)
                         if os.path.isdir(os.path.join(args.raw_root, d)))
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    overrides = {}
    for kv in args.template.split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            overrides[k.strip()] = v.strip()

    os.makedirs(args.out_root, exist_ok=True)
    print(f"building {len(targets)} target(s): {targets}")
    print(f"nproc={args.nproc} raid={args.raid} "
          f"cap_act={args.max_actives or 'all'} cap_ina={args.max_inactives or 'all'}")
    for i, t in enumerate(targets):
        print(f"\n[{i + 1}/{len(targets)}] {t}")
        build_target(t, args.raw_root, args.out_root, i, args.raid,
                     args.max_actives, args.max_inactives, args.nproc,
                     args.seed, overrides.get(t, ""))
    print("\ndone.")


if __name__ == "__main__":
    main()
