import sys, torch, torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore


def enc(tower, sample, pfx):
    dist = sample["net_input"][f"{pfx}_src_distance"]
    et = sample["net_input"][f"{pfx}_src_edge_type"]
    st = sample["net_input"][f"{pfx}_src_tokens"]
    pad = st.eq(tower.padding_idx)
    x = tower.embed_tokens(st)
    n = dist.size(-1)
    bias = tower.gbf_proj(tower.gbf(dist, et)).permute(0, 3, 1, 2).contiguous().view(-1, n, n)
    rep = tower.encoder(x, padding_mask=pad, attn_mask=bias)[0]   # (B, L, 512) per-atom
    return rep, pad


def strip(rep, pad):                 # drop padding, then BOS (first) and EOS (last)
    r = rep[~pad]
    return r[1:-1]


def main(args):
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

    ps = next(iter(torch.utils.data.DataLoader(poc, batch_size=1, collate_fn=poc.collater)))
    ms = next(iter(torch.utils.data.DataLoader(mol, batch_size=4, collate_fn=mol.collater)))
    if use_cuda:
        ps = unicore.utils.move_to_cuda(ps)
        ms = unicore.utils.move_to_cuda(ms)

    with torch.no_grad():
        P, Ppad = enc(model.pocket_model, ps, "pocket")
        L, Lpad = enc(model.mol_model, ms, "mol")
    print("pocket per-atom reps :", tuple(P.shape))
    print("molec. per-atom reps :", tuple(L.shape))

    p = strip(P[0], Ppad[0]).float()         # (M_pkt, 512)
    l = strip(L[0], Lpad[0]).float()         # (N_mol, 512)
    print("after strip -> pocket atoms:", p.shape[0], "| molecule-0 atoms:", l.shape[0])

    S = F.normalize(l, dim=-1) @ F.normalize(p, dim=-1).T   # (N_mol, M_pkt) cosine
    print("similarity matrix S  :", tuple(S.shape))
    print("RAW MaxSim (mean-of-max):", round(S.max(dim=1).values.mean().item(), 4))
    print("mol-atom -> best pocket-atom (first 10):", S.argmax(dim=1)[:10].tolist())


def cli():
    p = options.get_validation_parser()
    p.add_argument("--extract-target", type=str, default="ppara")
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
