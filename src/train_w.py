import os, random
import torch, numpy as np, torch.nn.functional as F
from unicore import checkpoint_utils, distributed_utils, options, tasks
import unicore
from tqdm import tqdm


def enc(tower, sample, pfx, n_layers):
    """Per-atom reps. n_layers=None/full -> final layer (+final LN), else partial forward up to layer k (no final LN)."""
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


_DS = {}   # open each LMDB once per process; lmdb refuses a second open of the same file


def get_ds(task, root, target):
    if target not in _DS:
        base = f"{root}/{target}"
        poc = task.load_pockets_dataset(base + "/pocket.lmdb")
        mol = task.load_mols_dataset(base + "/mols.lmdb", "atoms", "coordinates")
        _DS[target] = (poc, mol)
    return _DS[target]


@torch.no_grad()
def encode_target(task, model, root, target, use_cuda, max_decoys, layer, full):
    poc, mol = get_ds(task, root, target)
    need_cls = max_decoys is None   # eval mode needs the final-layer CLS for the DrugCLIP dot

    ps = next(iter(torch.utils.data.DataLoader(poc, 1, collate_fn=poc.collater)))
    if use_cuda:
        ps = unicore.utils.move_to_cuda(ps)
    Pa, Ppad = enc(model.pocket_model, ps, "pocket", layer)
    p_atoms = Pa[0][atom_mask_from_pad(Ppad)[0]].float()
    pcls = None
    if need_cls:
        Pf = Pa if layer >= full else enc(model.pocket_model, ps, "pocket", None)[0]
        pcls = F.normalize(model.pocket_project(Pf[0, 0:1]).float(), dim=-1)

    actives, decoys = [], []
    all_atoms, all_labels, all_cls = [], [], []
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
                all_atoms.append(ai); all_labels.append(int(labs[i])); all_cls.append(cls[i])
            elif labs[i] == 1:
                actives.append(ai)
            elif len(decoys) < max_decoys:
                decoys.append(ai)
        if (not need_cls) and len(decoys) >= max_decoys:
            break
    if need_cls:
        return p_atoms, pcls, all_atoms, np.array(all_labels), all_cls
    return p_atoms, actives, decoys


def maxsim(l, p, Wm, Wp):
    L = F.normalize(Wm(l), dim=-1)
    Pm = F.normalize(Wp(p), dim=-1)
    return (L @ Pm.T).max(dim=1).values.mean()


def raw_maxsim(l, p):
    return (F.normalize(l, dim=-1) @ F.normalize(p, dim=-1).T).max(dim=1).values.mean()


def run_layer(args, task, model, dev, use_cuda, full, train_ts, test_t, layer, cal_metrics):
    print(f"\n########## LAYER {layer}{'  (final + LN)' if layer >= full else ''} ##########")
    cache = {}
    for t in tqdm(train_ts, desc=f"encode train (layer {layer})"):
        p, acts, decs = encode_target(task, model, args.dude_root, t, use_cuda,
                                      args.decoys_per_target, layer, full)
        if len(acts) > 0 and len(decs) > 0:
            cache[t] = (p, acts, decs)
    ts = list(cache.keys())

    d = args.proj_dim
    Wm = torch.nn.Linear(512, d, bias=False).to(dev)
    Wp = torch.nn.Linear(512, d, bias=False).to(dev)
    tau = torch.nn.Parameter(torch.tensor(0.1, device=dev))
    opt = torch.optim.Adam(list(Wm.parameters()) + list(Wp.parameters()) + [tau], lr=1e-3)
    K = args.neg_per_step
    for step in range(args.steps):
        t = random.choice(ts)
        p, acts, decs = cache[t]
        a = random.choice(acts)
        ds = random.sample(decs, min(K, len(decs)))
        s = [maxsim(a, p, Wm, Wp)] + [maxsim(dd, p, Wm, Wp) for dd in ds]
        loss = -F.log_softmax(torch.stack(s) / tau.clamp(min=1e-2), dim=0)[0]
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 1000 == 0:
            print(f"  step {step+1:5d}  loss {loss.item():.4f}  tau {tau.item():.3f}")
    del cache; torch.cuda.empty_cache()

    p_atoms, pcls, atoms, labels, clss = encode_target(task, model, args.dude_root, test_t,
                                                       use_cuda, None, layer, full)
    with torch.no_grad():
        raw = np.array([raw_maxsim(a, p_atoms).item() for a in atoms])
        trW = np.array([maxsim(a, p_atoms, Wm, Wp).item() for a in atoms])
        dot = (F.normalize(torch.stack(clss), dim=-1) @ pcls.T).squeeze(1).cpu().numpy()

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    print(f"=== {test_t}: {int(labels.sum())} actives / {int((labels==0).sum())} decoys (held out) | maxsim layer {layer} ===")
    print(f"{'method':28s}  {'AUC':>7s}  {'BEDROC':>7s}  {'EF@0.5%':>8s}  {'EF@1%':>7s}")
    rows = [("DrugCLIP pooled dot", dot), ("RAW MaxSim (no W)", raw), ("Trained-W MaxSim", trW)]
    for a in [0.25, 0.5, 1.0]:
        rows.append((f"dot + {a:g}*maxsim", z(dot) + a * z(trW)))
    for name, sc in rows:
        auc, bedroc, ef, re = cal_metrics(labels, sc, 80.5)
        print(f"{name:28s}  {auc:7.4f}  {bedroc:7.4f}  {ef['0.005']:8.2f}  {ef['0.01']:7.2f}")
    del atoms, raw, trW, dot; torch.cuda.empty_cache()


def main(args):
    from unimol.tasks.drugclip import cal_metrics
    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(args.device_id)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args); model.load_state_dict(state["model"], strict=False)
    if use_cuda:
        model.cuda()
    model.eval()
    dev = next(model.parameters()).device
    full = len(model.mol_model.encoder.layers)

    targets = sorted(os.listdir(args.dude_root))
    test_t = args.test_target
    train_ts = [t for t in targets if t != test_t][: args.n_train]
    layers = [int(x) for x in args.layers.split(",")]
    print("train targets:", train_ts)
    print("test target  :", test_t, "| encoder layers:", full, "| sweeping:", layers)

    for layer in layers:
        run_layer(args, task, model, dev, use_cuda, full, train_ts, test_t, layer, cal_metrics)


def cli():
    p = options.get_validation_parser()
    p.add_argument("--dude-root", type=str, required=True)
    p.add_argument("--test-target", type=str, default="ppara")
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--decoys-per-target", type=int, default=1500)
    p.add_argument("--neg-per-step", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--layers", type=str, default="15,14,13")
    options.add_model_args(p)
    distributed_utils.call_main(options.parse_args_and_arch(p), main)


if __name__ == "__main__":
    cli()
