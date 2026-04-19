from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mutsig.cosmic import cosine_similarity_matrix
from mutsig.features import synthetic_mutation_matrix
from mutsig.nmf import NMF
from mutsig.plotting import plot_signature_panel
FIG_DIR = Path(__file__).resolve().parent.parent / 'figures'

def best_match_permutation(W: np.ndarray, W_true: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    sims = cosine_similarity_matrix(W, W_true)
    row, col = linear_sum_assignment(-sims)
    perm = np.argsort(col)
    return (W[:, perm], sims[row, col][np.argsort(col)])

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    K = 4
    N = 80
    V, W_true, H_true = synthetic_mutation_matrix(n_samples=N, n_signatures=K, avg_mutations=1500, seed=3)
    print(f'Synthetic V: {V.shape}, total mutations = {V.values.sum():.0f}')
    model = NMF(n_components=K, loss='kl', max_iter=2000, tol=1e-06, random_state=3)
    model.fit(V.values)
    print(f'Converged in {model.n_iter_} iterations (final KL = {model.loss_history_[-1]:.2f})')
    W_aligned, sims = best_match_permutation(model.W_, W_true)
    print('Per-signature cosine similarities to ground truth:')
    for k, s in enumerate(sims):
        print(f'  sig {k}: {s:.3f}')
    fig, axes = plt.subplots(2 * K, 1, figsize=(12, 2.5 * 2 * K))
    from mutsig.plotting import plot_signature
    for k in range(K):
        plot_signature(W_true[:, k], title=f'Ground truth signature {k}', ax=axes[2 * k])
        plot_signature(W_aligned[:, k], title=f'Recovered signature {k} (cos={sims[k]:.3f})', ax=axes[2 * k + 1])
    fig.tight_layout()
    out = FIG_DIR / 'synthetic_recovery.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'saved {out.relative_to(FIG_DIR.parent)}')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(model.loss_history_)
    ax.set_xlabel('iteration')
    ax.set_ylabel('KL divergence')
    ax.set_title('NMF convergence (synthetic, K=4)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / 'synthetic_loss_curve.png'
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f'saved {out.relative_to(FIG_DIR.parent)}')
if __name__ == '__main__':
    main()
