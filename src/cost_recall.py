import os, time, random
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
    m = nonpad.clone(); m[:, 0] = False
    lens = nonpad.sum(1)
    m[torch.arange(pad.size(0), device=pad.device), lens - 1] = False
    return m


_DS = {}
def get_ds(task, root, target):
    if target not in _DS:
        base = f"{root}/{target}"
        poc = task.load_pockets_dataset(base + "/pocket.lmdb")
        mol = task.load_mols_dataset(base + "/mols.lmdb", "atoms", "coordinates")
        _DS[target] = (poc, mol)
    return _DS[target]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args):
    from unimol.tasks.drugclip import cal_metrics  # noqa: F401 (kept parallel to eval script)
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
    d = args.proj_dim
    Wm = torch.nn.Linear(512, d, bias=False).to(dev)   # untrained: MaxSim COST is W-independent
    Wp = torch.nn.Linear(512, d, bias=False).to(dev)
    Wm.eval(); Wp.eval()

    targets = sorted(os.listdir(args.dude_root))[: args.n_test]
    Ks = [int(k) for k in args.ks.split(",")]

    recall = {k: [] for k in Ks}        # Recall@K per target
    enc_per_mol, maxsim_per_k, dot_time, store = [], [], [], []
    n_meas = 0

    for ti, t in enumerate(tqdm(targets, desc="targets")):
        poc, mol = get_ds(task, args.dude_root, t)
        ps = next(iter(torch.utils.data.DataLoader(poc, 1, collate_fn=poc.collater)))
        if use_cuda:
            ps = unicore.utils.move_to_cuda(ps)
        with torch.no_grad():
            P, Ppad = enc(model.pocket_model, ps, "pocket", layer)
            p_atoms = F.normalize(Wp(P[0][atom_mask_from_pad(Ppad)[0]].float()), dim=-1)
            pcls = F.normalize(model.pocket_project(
                (P if layer >= full else enc(model.pocket_model, ps, "pocket", None)[0])[0, 0:1]).float(), dim=-1)

        mcls_all, labels_all, atomcounts, atoms_keep = [], [], [], []
        measure = ti < args.n_measure
        t_enc = 0.0; n_mol = 0
        loader = torch.utils.data.DataLoader(mol, 64, collate_fn=mol.collater)
        with torch.no_grad():
            for samp in loader:
                if use_cuda:
                    samp = unicore.utils.move_to_cuda(samp)
                sync(); t0 = time.perf_counter()
                La, Lpad = enc(model.mol_model, samp, "mol", layer)
                Lf = La if layer >= full else enc(model.mol_model, samp, "mol", None)[0]
                sync(); t_enc += time.perf_counter() - t0
                amask = atom_mask_from_pad(Lpad)
                mcls_all.append(F.normalize(model.mol_project(Lf[:, 0, :]).float(), dim=-1))
                labs = samp["target"].detach().cpu().numpy()
                labels_all.extend(int(x) for x in labs)
                for i in range(La.size(0)):
                    na = int(amask[i].sum())
                    atomcounts.append(na)
                    if measure and len(atoms_keep) < max(Ks):
                        atoms_keep.append(F.normalize(Wm(La[i][amask[i]].float()), dim=-1))
                n_mol += La.size(0)

        mcls = torch.cat(mcls_all, 0)                     # (N, 128)
        labels = np.array(labels_all, dtype=np.int32)
        A = int(labels.sum())

        # Stage-1 online cost: the dot screen (one matmul over precomputed CLS)
        sync(); t0 = time.perf_counter()
        dot = (mcls @ pcls.T).squeeze(1)
        order = torch.argsort(dot, descending=True).cpu().numpy()
        sync(); t_dot = time.perf_counter() - t0

        # Recall@K from the dot ranking
        ranked_lab = labels[order]
        for k in Ks:
            recall[k].append(float(ranked_lab[:k].sum()) / max(A, 1))

        # storage per molecule: stored CLS (128-d) vs hypothetical per-atom tokens (avg_atoms x 512)
        avg_atoms = float(np.mean(atomcounts))
        cls_bytes = d * 4
        atom_bytes = avg_atoms * 512 * 4
        store.append((avg_atoms, cls_bytes, atom_bytes))

        if measure:
            enc_per_mol.append(t_enc / max(n_mol, 1))
            dot_time.append(t_dot)
            # Stage-2 MaxSim cost for the shortlist (per-mol mean-of-max against pocket atoms)
            kmax = min(max(Ks), len(atoms_keep))
            sync(); t0 = time.perf_counter()
            for a in atoms_keep[:kmax]:
                _ = (a @ p_atoms.T).max(dim=1).values.mean()
            sync(); t_ms = (time.perf_counter() - t0) / max(kmax, 1)
            maxsim_per_k.append(t_ms)
            atoms_keep = []
            n_meas += 1
        torch.cuda.empty_cache()

    # ---------- report ----------
    print(f"\n=== Recall@K of DrugCLIP dot retrieval (mean over {len(targets)} targets, layer {layer}) ===")
    print(f"{'K':>6s}  {'Recall@K':>10s}")
    for k in Ks:
        print(f"{k:6d}  {np.mean(recall[k]):10.4f}")
    print("  (Stage-2 reranks only the top-K, so Recall@K is the end-to-end ceiling.)")

    epm = float(np.mean(enc_per_mol)); mpk = float(np.mean(maxsim_per_k)); dt = float(np.mean(dot_time))
    avg_atoms = float(np.mean([s[0] for s in store]))
    cls_b = float(np.mean([s[1] for s in store])); atom_b = float(np.mean([s[2] for s in store]))
    print(f"\n=== Measured cost ({n_meas} targets, {'GPU' if use_cuda else 'CPU'}, batch 64, layer {layer}) ===")
    print(f"  per-molecule encode (forward)      : {epm*1e3:8.3f} ms")
    print(f"  per-molecule MaxSim vs pocket      : {mpk*1e3:8.3f} ms   (unoptimised python loop)")
    print(f"  Stage-1 dot screen over all mols   : {dt*1e3:8.3f} ms / target (precomputed CLS)")
    for k in [100, 1000]:
        overhead = k * (epm + mpk)
        print(f"  Stage-2 overhead @K={k:<4d}            : {overhead*1e3:8.1f} ms / target  (re-encode K + MaxSim)")
    print(f"\n=== Storage per molecule (avg {avg_atoms:.1f} atoms) ===")
    print(f"  stored DrugCLIP CLS  ({d}-d)        : {cls_b:8.0f} bytes")
    print(f"  per-atom tokens (avg_atoms x 512)  : {atom_b:8.0f} bytes   = {atom_b/cls_b:5.1f}x the CLS")
    print(f"  -> storing per-atom for a 500M library is ~{atom_b*5e8/1e12:.1f} TB vs {cls_b*5e8/1e12:.3f} TB; hence re-encode Top-K.")


def cli():
    p = options.get_validation_parser()
    p.add_argument("--dude-root", type=str, required=True)
    p.add_argument("--n-test", type=int, default=20)       # targets for Recall@K
    p.add_argument("--n-measure", type=int, default=3)     # targets for timing
    p.add_argument("--ks", type=str, default="50,100,200,500,1000")
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--layer", type=int, default=12)
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
