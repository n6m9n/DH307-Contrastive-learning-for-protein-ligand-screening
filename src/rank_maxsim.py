import torch, numpy as np, torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore
from tqdm import tqdm


def enc(tower, sample, pfx):
    dist = sample["net_input"][f"{pfx}_src_distance"]
    et = sample["net_input"][f"{pfx}_src_edge_type"]
    st = sample["net_input"][f"{pfx}_src_tokens"]
    pad = st.eq(tower.padding_idx)
    x = tower.embed_tokens(st)
    n = dist.size(-1)
    bias = tower.gbf_proj(tower.gbf(dist, et)).permute(0, 3, 1, 2).contiguous().view(-1, n, n)
    rep = tower.encoder(x, padding_mask=pad, attn_mask=bias)[0]
    return rep, pad


def atom_mask_from_pad(pad):
    # pad: (B,L) True=padding. Real atoms = non-pad, minus BOS (index 0) and EOS (last non-pad).
    nonpad = ~pad
    m = nonpad.clone()
    m[:, 0] = False
    lens = nonpad.sum(1)
    m[torch.arange(pad.size(0), device=pad.device), lens - 1] = False
    return m


def main(args):
    from unimol.tasks.drugclip import cal_metrics   # import after unicore loads --user-dir
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)
    if use_cuda:
        model.cuda()
    model.eval()

    base = "./data/DUD-E/raw/all/" + args.extract_target
    poc = task.load_pockets_dataset(base + "/pocket.lmdb")
    mol = task.load_mols_dataset(base + "/mols.lmdb", "atoms", "coordinates")

    # ---- encode the pocket once ----
    ps = next(iter(torch.utils.data.DataLoader(poc, batch_size=1, collate_fn=poc.collater)))
    if use_cuda:
        ps = unicore.utils.move_to_cuda(ps)
    with torch.no_grad():
        P, Ppad = enc(model.pocket_model, ps, "pocket")
        pmask = atom_mask_from_pad(Ppad)
        p_atoms = F.normalize(P[0][pmask[0]].float(), dim=-1)                 # (M,512) per-atom
        p_cls = F.normalize(model.pocket_project(P[0, 0:1]).float(), dim=-1)  # (1,128) pooled

    maxsim_scores, dot_scores, labels = [], [], []
    loader = torch.utils.data.DataLoader(mol, batch_size=64, collate_fn=mol.collater)
    for samp in tqdm(loader):
        if use_cuda:
            samp = unicore.utils.move_to_cuda(samp)
        with torch.no_grad():
            L, Lpad = enc(model.mol_model, samp, "mol")                      # (B,Lm,512)
            amask = atom_mask_from_pad(Lpad).float()                         # (B,Lm)
            Ln = F.normalize(L.float(), dim=-1)
            S = torch.einsum("bld,md->blm", Ln, p_atoms)                     # (B,Lm,M)
            best = S.max(dim=-1).values                                      # (B,Lm)
            ms = (best * amask).sum(1) / amask.sum(1).clamp(min=1)           # (B,) MaxSim
            mcls = F.normalize(model.mol_project(L[:, 0, :]).float(), dim=-1)  # (B,128)
            dot = (mcls @ p_cls.T).squeeze(1)                               # (B,) pooled
        maxsim_scores.append(ms.cpu().numpy())
        dot_scores.append(dot.cpu().numpy())
        labels.extend(samp["target"].detach().cpu().numpy())

    maxsim_scores = np.concatenate(maxsim_scores)
    dot_scores = np.concatenate(dot_scores)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n{args.extract_target}: {int(labels.sum())} actives / {int((labels == 0).sum())} decoys\n")
    print(f"{'method':24s}  {'AUC':>7s}  {'BEDROC':>7s}  {'EF@0.5%':>8s}  {'EF@1%':>7s}")
    for name, sc in [("DrugCLIP pooled dot", dot_scores), ("RAW MaxSim (no W)", maxsim_scores)]:
        auc, bedroc, ef, re = cal_metrics(labels, sc, 80.5)
        print(f"{name:24s}  {auc:7.4f}  {bedroc:7.4f}  {ef['0.005']:8.2f}  {ef['0.01']:7.2f}")


def cli():
    p = options.get_validation_parser()
    p.add_argument("--extract-target", type=str, default="ppara")
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
