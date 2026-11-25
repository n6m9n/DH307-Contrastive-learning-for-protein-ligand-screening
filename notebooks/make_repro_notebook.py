"""Generate repro_one_target.ipynb -- a one-target, end-to-end reproduction of the
DrugCLIP MaxSim reranker: frozen checkpoint -> trained projection W -> Stage-1 dot
shortlist -> Stage-2 MaxSim rerank -> final cascade ranking + enrichment. Runs on the
lab PC (env drugclip) from ~/drugclip/DrugCLIP. Writes the notebook next to this file."""
import json, os

TARGET = "akt1"   # DUD-E held-out test target with a clear reranking gain

def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                       "outputs": [], "source": src}

cells = []

cells.append(md(
"# Reproducing one target end-to-end: DrugCLIP + MaxSim reranker\n"
"\n"
"This notebook takes a single held-out DUD-E target (`%s`) from DrugCLIP's shortlist\n"
"through to the final reranked ordering, using the **frozen** DrugCLIP checkpoint and a\n"
"**trained linear projection** W.\n"
"\n"
"Pipeline: load checkpoint -> train/load W (layer 12) -> Stage-1 dot ranking ->\n"
"top-1000 shortlist -> Stage-2 MaxSim rerank -> cascade (rerank shortlist, rest keep dot\n"
"order) -> enrichment.\n"
"\n"
"Run from `~/drugclip/DrugCLIP` in the `drugclip` conda env.\n" % TARGET))

cells.append(md("## 1. Load the frozen DrugCLIP model"))
cells.append(code(
"import os, sys, random\n"
"import numpy as np, torch\n"
"import torch.nn.functional as F\n"
"from unicore import options, tasks, checkpoint_utils, distributed_utils\n"
"import unicore\n"
"\n"
"# Build the unicore args exactly as the CLI scripts do, then add the W hyperparameters.\n"
"parser = options.get_validation_parser()\n"
"parser.add_argument('--dude-root', type=str,\n"
"    default=os.path.expanduser('~/drugclip/dude_data/data/protein/DUD-E/raw/all'))\n"
"options.add_model_args(parser)\n"
"argv = ['--user-dir','./unimol','./data','--valid-subset','test','--task','drugclip',\n"
"        '--loss','in_batch_softmax','--arch','drugclip','--seed','1',\n"
"        '--path','../checkpoint_best.pt','--max-pocket-atoms','511',\n"
"        '--batch-size','8','--num-workers','2','--ddp-backend=c10d']\n"
"sys.argv = ['notebook'] + argv\n"
"args = options.parse_args_and_arch(parser)\n"
"for k, v in dict(decoys_per_target=800, neg_per_step=32, hard_pool=96, steps=4000,\n"
"                 proj_dim=128, dropout=0.1, proj_wd=1e-2, tau=0.05,\n"
"                 share_w=False, random_neg=False, n_test=20, n_val=10, n_train=60).items():\n"
"    setattr(args, k, v)\n"
"\n"
"random.seed(1); np.random.seed(1); torch.manual_seed(1)\n"
"state = checkpoint_utils.load_checkpoint_to_cpu(args.path)\n"
"task = tasks.setup_task(args)\n"
"model = task.build_model(args); model.load_state_dict(state['model'], strict=False)\n"
"use_cuda = torch.cuda.is_available()\n"
"if use_cuda: model.cuda()\n"
"model.eval()\n"
"dev = next(model.parameters()).device\n"
"full = len(model.mol_model.encoder.layers)\n"
"LAYER, ALPHA = 12, 0.5\n"
"print('model loaded on', dev, '| layers', full, '| token layer', LAYER)"))

cells.append(md(
"## 2. The trained projection W\n"
"W (two linear maps, 512->128) is the only trained part. We load a cached copy if\n"
"present, otherwise train it on the 60 DUD-E training targets (seed 1, ~10-15 min) and\n"
"cache it. The encoders stay frozen throughout."))
cells.append(code(
"from rerank_eval import train_W, encode_target, maxsim\n"
"\n"
"targets = sorted(os.listdir(args.dude_root))\n"
"train_ts = targets[args.n_test + args.n_val: args.n_test + args.n_val + args.n_train]\n"
"W_PATH = os.path.expanduser('~/W_layer12.pt')\n"
"\n"
"Wm = torch.nn.Linear(512, 128, bias=False).to(dev)\n"
"Wp = torch.nn.Linear(512, 128, bias=False).to(dev)\n"
"if os.path.exists(W_PATH):\n"
"    sd = torch.load(W_PATH, map_location=dev)\n"
"    Wm.load_state_dict(sd['Wm']); Wp.load_state_dict(sd['Wp'])\n"
"    print('loaded cached W from', W_PATH)\n"
"else:\n"
"    print('training W on', len(train_ts), 'DUD-E targets ...')\n"
"    Wm, Wp = train_W(args, task, model, dev, use_cuda, full, LAYER, train_ts)\n"
"    torch.save({'Wm': Wm.state_dict(), 'Wp': Wp.state_dict()}, W_PATH)\n"
"    print('saved W to', W_PATH)\n"
"Wm.eval(); Wp.eval()"))

cells.append(md(
"## 3. Stage 1 - DrugCLIP dot ranking and the shortlist\n"
"Encode the target's pocket and every library molecule, score each by the pooled\n"
"dot product (DrugCLIP), and keep the top-1000 as the shortlist."))
cells.append(code(
"TARGET = %r\n"
"p_atoms, pcls, atoms, labels, clss = encode_target(task, model, args.dude_root,\n"
"                                                   TARGET, use_cuda, None, LAYER, full)\n"
"labels = np.asarray(labels)\n"
"dot = (F.normalize(torch.stack(clss), dim=-1) @ pcls.T).squeeze(1).cpu().numpy()\n"
"n_mols, n_act = len(dot), int(labels.sum())\n"
"K = 1000\n"
"order_dot = np.argsort(-dot, kind='mergesort')\n"
"short = order_dot[:K]\n"
"print(f'{TARGET}: {n_mols} molecules, {n_act} actives ({100*n_act/n_mols:.2f}%%)')\n"
"print(f'shortlist (top-{K} by dot) contains {int(labels[short].sum())} actives '\n"
"      f'-> Recall@{K} = {labels[short].sum()/n_act:.3f} (ceiling for any reranker)')" % TARGET))

cells.append(md(
"## 4. Stage 2 - MaxSim rerank of the shortlist\n"
"For each shortlisted molecule, compute the trained MaxSim score against the pocket,\n"
"combine with the dot via `z(dot) + alpha * z(maxsim)`, and rerank. Molecules below the\n"
"shortlist keep their dot order (the deployment cascade)."))
cells.append(code(
"def z(x):\n"
"    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-8)\n"
"\n"
"ms = np.array([maxsim(atoms[i].to(dev), p_atoms, Wm, Wp).item() for i in short])\n"
"combined = z(dot[short]) + ALPHA * z(ms)              # rerank score on the shortlist\n"
"short_reranked = short[np.argsort(-combined, kind='mergesort')]\n"
"final_order = np.concatenate([short_reranked, order_dot[K:]])  # cascade\n"
"print('reranked the', K, 'shortlisted molecules; built final cascade ranking')"))

cells.append(md(
"## 5. Enrichment: DrugCLIP dot vs the reranked cascade\n"
"Early-recognition metrics on the full ranking, plus actives recovered in the very top."))
cells.append(code(
"from unimol.tasks.drugclip import cal_metrics\n"
"\n"
"def rank_score(order):\n"
"    s = np.empty(len(order)); s[order] = np.arange(len(order), 0, -1); return s\n"
"\n"
"def report(name, order):\n"
"    s = rank_score(order)\n"
"    auc, bed, ef, _ = cal_metrics(labels, s, 80.5)\n"
"    top = order[:100]\n"
"    a50 = int(labels[order[:50]].sum()); a100 = int(labels[top].sum())\n"
"    print(f'{name:16s} BEDROC {bed:.4f}  EF@0.5%% {ef[\"0.005\"]:6.2f}  '\n"
"          f'EF@1%% {ef[\"0.01\"]:6.2f}  actives@50 {a50:3d}  actives@100 {a100:3d}')\n"
"    return bed, ef['0.005'], a50, a100\n"
"\n"
"print('target:', TARGET, '\\n')\n"
"b0 = report('DrugCLIP dot', order_dot)\n"
"b1 = report('dot + MaxSim', final_order)\n"
"print(f'\\nΔBEDROC {b1[0]-b0[0]:+.4f}   ΔEF@0.5%% {b1[1]-b0[1]:+.2f}   '\n"
"      f'Δactives@50 {b1[2]-b0[2]:+d}   Δactives@100 {b1[3]-b0[3]:+d}')"))

cells.append(md("## 6. Cumulative actives in the top of the ranking (optional plot)"))
cells.append(code(
"N = 1000\n"
"cum_dot = np.cumsum(labels[order_dot[:N]])\n"
"cum_rr  = np.cumsum(labels[final_order[:N]])\n"
"print(f'actives recovered by rank {N}:  dot {cum_dot[-1]}   reranked {cum_rr[-1]}')\n"
"try:\n"
"    import matplotlib; matplotlib.use('Agg')\n"
"    import matplotlib.pyplot as plt\n"
"    plt.figure(figsize=(6,4))\n"
"    plt.plot(range(1, N+1), cum_dot, label='DrugCLIP dot', lw=2)\n"
"    plt.plot(range(1, N+1), cum_rr,  label='dot + MaxSim (reranked)', lw=2)\n"
"    plt.xlabel('rank'); plt.ylabel('cumulative actives recovered')\n"
"    plt.title(f'{TARGET}: early recovery, dot vs reranked'); plt.legend(); plt.tight_layout()\n"
"    plt.savefig(os.path.expanduser('~/repro_%s_recovery.png'), dpi=120)\n"
"    print('saved plot to ~/repro_%s_recovery.png'); plt.show()\n"
"except ModuleNotFoundError:\n"
"    print('(matplotlib not installed - skipping plot; the counts above are the result)')" % (TARGET, TARGET)))

cells.append(md(
"## 7. Takeaways\n"
"- The trained MaxSim reranker improves early enrichment on this held-out target at\n"
"  negligible added cost (MaxSim ~0.013 ms/mol; the real cost is re-encoding the shortlist).\n"
"- End-to-end recovery is capped by the Stage-1 Recall@1000 shown above.\n"
"- **Scope note.** This notebook demonstrates the *shortlist cascade*: Stage 2 reranks only\n"
"  the top-1000 and z-scores within it, so the counts here are the cascade-bounded version.\n"
"  The paper's full-library rerank (Table 1 and Appendix C) gives a slightly larger per-target\n"
"  gain because it can also promote actives from below the shortlist; e.g. for akt1 this\n"
"  notebook reports actives@50 12->27, while the full-library Table gives 12->31.\n"
"- **Caveat (specificity):** the companion `pocket_subst.py` control shows that on DUD-E\n"
"  this reranking gain is largely *not pocket-specific* (a wrong pocket, and a ligand-only\n"
"  head, rerank about as well), consistent with DUD-E's known decoy bias. Swap the pocket\n"
"  in Stage 2 (use another target's `p_atoms`) to see this directly."))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repro_one_target.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
