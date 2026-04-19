from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mutsig.cosmic import cosine_similarity_matrix, load_cosmic_sbs, match_to_cosmic
from mutsig.download import fetch_cosmic_sbs, fetch_project_mafs, load_project_mafs
from mutsig.features import SBS96_CHANNELS, maf_to_sbs96
from mutsig.nmf import NMF
from mutsig.plotting import plot_signature

def build_project_matrix(project_id: str, data_dir: Path, n_cases: int) -> pd.DataFrame:
    maf_paths = fetch_project_mafs(project_id, data_dir / 'raw' / project_id, limit=n_cases)
    print(f'[{project_id}] parsing {len(maf_paths)} MAFs...')
    mutations = load_project_mafs(maf_paths)
    print(f"[{project_id}] {len(mutations)} SNVs across {mutations['Tumor_Sample_Barcode'].nunique()} samples")
    V = maf_to_sbs96(mutations)
    out = data_dir / 'processed' / f'{project_id}_SBS96.tsv'
    out.parent.mkdir(parents=True, exist_ok=True)
    V.to_csv(out, sep='\t')
    print(f'[{project_id}] saved SBS-96 matrix to {out}')
    return V

def fit_nmf_sweep(V: np.ndarray, k_values: list[int], n_restarts: int=5, max_iter: int=1500) -> dict[int, dict]:
    results = {}
    for K in k_values:
        best = None
        for seed in range(n_restarts):
            model = NMF(n_components=K, loss='kl', max_iter=max_iter, tol=1e-05, init='nndsvd' if seed == 0 else 'random', random_state=seed)
            model.fit(V)
            loss = model.loss_history_[-1]
            if best is None or loss < best['loss']:
                best = {'model': model, 'loss': loss, 'seed': seed}
        results[K] = best
        print(f"  K={K}: best KL = {best['loss']:.2f} (seed {best['seed']})")
    return results

def plot_and_save_signatures(W: np.ndarray, labels: list[str], out_path: Path):
    K = W.shape[1]
    fig, axes = plt.subplots(K, 1, figsize=(12, 2.8 * K), squeeze=False)
    axes = axes.ravel()
    for k in range(K):
        plot_signature(W[:, k], title=labels[k], ax=axes[k])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def run(project_id: str, n_cases: int, k_values: list[int], out_root: Path) -> dict:
    data_dir = out_root / 'data'
    fig_dir = out_root / 'figures'
    V = build_project_matrix(project_id, data_dir, n_cases)
    V_arr = V.values.astype(np.float64)
    print(f'\n[{project_id}] fitting NMF for K in {k_values}...')
    sweep = fit_nmf_sweep(V_arr, k_values)
    cosmic_path = data_dir / 'cosmic' / 'COSMIC_v3.3_SBS_GRCh38.txt'
    cosmic_path = fetch_cosmic_sbs(cosmic_path)
    cosmic = load_cosmic_sbs(cosmic_path)
    print(f'loaded COSMIC v3.3 with {cosmic.shape[1]} signatures')
    summary = {}
    for K, entry in sweep.items():
        W = entry['model'].W_
        assignment = match_to_cosmic(W, cosmic, min_similarity=0.85)
        labels = [f'Recovered sig {row.recovered_k}  ->  {row.cosmic_match}  (cos={row.cosine_similarity:.3f})' for _, row in assignment.iterrows()]
        out = fig_dir / f'{project_id}_K{K}_signatures.png'
        plot_and_save_signatures(W, labels, out)
        print(f'[{project_id}] K={K}: saved {out.name}')
        summary[K] = {'loss': entry['loss'], 'assignment': assignment, 'fig': str(out.relative_to(out_root))}
    losses = [sweep[K]['loss'] for K in k_values]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, losses, marker='o')
    ax.set_xlabel('K (number of signatures)')
    ax.set_ylabel('KL divergence (best restart)')
    ax.set_title(f'{project_id}: reconstruction vs model order')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    loss_out = fig_dir / f'{project_id}_loss_vs_K.png'
    fig.savefig(loss_out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    summary['loss_curve_fig'] = str(loss_out.relative_to(out_root))
    return summary
