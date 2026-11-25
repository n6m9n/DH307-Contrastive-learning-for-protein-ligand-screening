"""
plip_gt.py -- compute PLIP TYPED-interaction ground truth for the 35 held-out holo
complexes, as a secondary analysis to the geometric 4 A contact test.

Runs in a SEPARATE conda env ('plip': openbabel + plip), NOT the drugclip env, so
the fragile OpenBabel install never touches the pinned torch/unimol stack. It only
produces a JSON of typed interactions; the encoding / W-projection / AUROC scoring
stays in contact_recover.py (drugclip env), which maps these interactions back to
its encoder tokens by crystal coordinate.

For each complex it builds a single-complex PDB (full receptor ATOM records + the
crystal ligand as HETATM 'LIG'), runs the PLIP CLI, and parses report.xml for every
interaction that carries both <ligcoo> and <protcoo> (hydrophobic, H-bond, pi-stack,
pi-cation, salt bridge, halogen, water bridge, metal). Each is stored as
{type, lig:[x,y,z], prot:[x,y,z]} in the SAME crystal frame the encoder tokens use.

Complex selection is identical to contact_recover.build_complexes:
  * DUD-E: sorted(dude_root)[:n_test] -> receptor.pdb + crystal_ligand.mol2, key "dude_<t>"
  * LIT-PCBA: sorted(litpcba_data) dirs with a raw dir -> alphabetical-first
    <pdbid>_protein.mol2 + <pdbid>_ligand.mol2, key "ltp_<t>"

Usage (lab PC, env plip, from ~/drugclip/DrugCLIP):
  ~/miniconda3/envs/plip/bin/python plip_gt.py \
    --dude-root ~/drugclip/dude_data/data/protein/DUD-E/raw/all --n-test 20 \
    --litpcba-data ~/drugclip/litpcba_data --litpcba-raw ~/litpcba_raw \
    --out ~/plip_gt.json 2>&1 | tee ~/plip_gt.log
"""
import os, sys, json, glob, argparse, subprocess, tempfile, shutil
import xml.etree.ElementTree as ET

# obabel/plip live in the same env bin as this interpreter, which may not be on PATH
_BIN = os.path.dirname(os.path.abspath(sys.executable))


def _exe(name):
    cand = os.path.join(_BIN, name)
    return cand if os.path.exists(cand) else name


# ---------- complex selection (mirrors contact_recover.build_complexes) ----------
def pick_litpcba_template(raw_tdir):
    prots = sorted(f for f in os.listdir(raw_tdir) if f.endswith("_protein.mol2"))
    pdbid = prots[0].split("_")[0]          # alphabetical-first == build_litpcba's choice
    return (os.path.join(raw_tdir, f"{pdbid}_protein.mol2"),
            os.path.join(raw_tdir, f"{pdbid}_ligand.mol2"), pdbid)


def list_complexes(args):
    """Return [(name, protein_path, ligand_path, protein_is_pdb)]."""
    comps = []
    dude_targets = sorted(os.listdir(args.dude_root))
    for t in dude_targets[:args.n_test]:
        rec = os.path.join(args.dude_root, t, "receptor.pdb")
        lig = os.path.join(args.dude_root, t, "crystal_ligand.mol2")
        if os.path.exists(rec) and os.path.exists(lig):
            comps.append((f"dude_{t}", rec, lig, True))
    if args.litpcba_data and args.litpcba_raw:
        for t in sorted(os.listdir(args.litpcba_data)):
            raw_tdir = os.path.join(args.litpcba_raw, t)
            if not os.path.isdir(raw_tdir):
                continue
            prot, lig, _ = pick_litpcba_template(raw_tdir)
            if os.path.exists(prot) and os.path.exists(lig):
                comps.append((f"ltp_{t}", prot, lig, False))
    return comps


# ---------- build a single-complex PDB (receptor ATOM + ligand HETATM 'LIG') ----------
def obabel_to_pdb(src, dst):
    r = subprocess.run([_exe("obabel"), src, "-O", dst], capture_output=True, text=True)
    if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
        raise RuntimeError(f"obabel failed on {src}: {r.stderr.strip()[:300]}")


def protein_atom_lines(protein_path, is_pdb, tmpd):
    if is_pdb:
        pdb = protein_path
    else:
        pdb = os.path.join(tmpd, "protein.pdb")
        obabel_to_pdb(protein_path, pdb)
    out = []
    with open(pdb) as f:
        for ln in f:
            if ln.startswith("ATOM"):          # protein backbone/side-chain only
                out.append(ln.rstrip("\n"))
    return out


def ligand_hetatm_lines(ligand_mol2, tmpd):
    lig_pdb = os.path.join(tmpd, "lig.pdb")
    obabel_to_pdb(ligand_mol2, lig_pdb)
    out = []
    with open(lig_pdb) as f:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                ln = ln.rstrip("\n").ljust(80)
                # force HETATM / resName LIG / chain Z / resSeq 999, keep coords+element
                new = ("HETATM" + ln[6:17] + "LIG" + ln[20:21] + "Z" +
                       "%4d" % 999 + ln[26:])
                out.append(new.rstrip())
    return out


def build_complex_pdb(name, protein_path, ligand_mol2, is_pdb, tmpd):
    prot = protein_atom_lines(protein_path, is_pdb, tmpd)
    lig = ligand_hetatm_lines(ligand_mol2, tmpd)
    if not prot or not lig:
        raise RuntimeError(f"empty protein ({len(prot)}) or ligand ({len(lig)})")
    dst = os.path.join(tmpd, "complex.pdb")
    with open(dst, "w") as f:
        f.write("\n".join(prot) + "\nTER\n")
        f.write("\n".join(lig) + "\nEND\n")
    return dst


# ---------- run PLIP + parse report.xml ----------
# PLIP ligcoo/protcoo are the genuine interacting ATOMS for these types (explicit
# atom-id fields: ligcarbonidx/protcarbonidx, donoridx/acceptoridx, don_idx/acc_idx)
# but are group CENTROIDS (ring/charge centers, members in *_idx_list) for these;
# water bridges are water-mediated. Only "atom" level gives true atom-pair recovery.
LEVEL = {
    "hydrophobic_interactions": "atom",
    "hydrogen_bonds": "atom",
    "halogen_bonds": "atom",
    "salt_bridges": "group",
    "pi_stacks": "group",
    "pi_cation_interactions": "group",
    "water_bridges": "water",
    "metal_complexes": "metal",
}


def _xyz(node):
    return [float(node.find("x").text), float(node.find("y").text),
            float(node.find("z").text)]


def run_plip(complex_pdb, tmpd):
    outd = os.path.join(tmpd, "plip_out")
    os.makedirs(outd, exist_ok=True)
    env = dict(os.environ)
    r = subprocess.run([_exe("plip"), "-f", complex_pdb, "-o", outd, "-x", "-q"],
                       capture_output=True, text=True, env=env)
    # PLIP 3.x writes <basename>_report.xml (not report.xml)
    cands = glob.glob(os.path.join(outd, "*report.xml"))
    if not cands:
        raise RuntimeError(f"no report xml: {r.stderr.strip()[-400:] or r.stdout.strip()[-400:]}")
    xml = cands[0]
    inters = []
    root = ET.parse(xml).getroot()
    for bs in root.findall(".//bindingsite"):
        hetid = bs.find("identifiers/hetid")
        if hetid is not None and hetid.text not in (None, "LIG"):
            continue
        ig = bs.find("interactions")
        if ig is None:
            continue
        for group in ig:                       # hydrophobic_interactions, hydrogen_bonds, ...
            gtype = group.tag
            for it in group:                   # each individual interaction
                lc, pc = it.find("ligcoo"), it.find("protcoo")
                if lc is None or pc is None:
                    continue
                try:
                    inters.append({"type": gtype, "level": LEVEL.get(gtype, "other"),
                                   "lig": _xyz(lc), "prot": _xyz(pc)})
                except (AttributeError, TypeError):
                    continue
    return inters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dude-root", required=True)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--litpcba-data", default="")
    ap.add_argument("--litpcba-raw", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    comps = list_complexes(args)
    print(f"selected {len(comps)} complexes "
          f"({sum(c[0].startswith('dude_') for c in comps)} DUD-E + "
          f"{sum(c[0].startswith('ltp_') for c in comps)} LIT-PCBA)", flush=True)

    gt, summary = {}, []
    for name, prot, lig, is_pdb in comps:
        tmpd = tempfile.mkdtemp(prefix="plipgt_")
        try:
            cpx = build_complex_pdb(name, prot, lig, is_pdb, tmpd)
            inters = run_plip(cpx, tmpd)
            gt[name] = inters
            by = {}
            for it in inters:
                by[it["type"]] = by.get(it["type"], 0) + 1
            summary.append((name, len(inters), by))
            print(f"  [ok]   {name:22s} {len(inters):3d} interactions  "
                  f"{ {k.replace('_interactions','').replace('_',''):v for k,v in by.items()} }",
                  flush=True)
        except Exception as e:
            gt[name] = []
            print(f"  [FAIL] {name:22s} {e}", flush=True)
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)

    with open(args.out, "w") as f:
        json.dump(gt, f)
    tot = sum(len(v) for v in gt.values())
    ok = sum(1 for v in gt.values() if v)
    print(f"\n[wrote {args.out}: {ok}/{len(comps)} complexes with >=1 interaction, "
          f"{tot} interactions total]")


if __name__ == "__main__":
    main()
