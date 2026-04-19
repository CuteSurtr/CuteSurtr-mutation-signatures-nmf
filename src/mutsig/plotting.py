from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from mutsig.features import SBS96_CHANNELS, SUBSTITUTIONS
SUBSTITUTION_COLORS = {('C', 'A'): '#03bcee', ('C', 'G'): '#010101', ('C', 'T'): '#e32926', ('T', 'A'): '#cac9c9', ('T', 'C'): '#a1ce63', ('T', 'G'): '#ebc6c4'}

def plot_signature(signature: np.ndarray, title: str='', ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    colors = []
    for ref, alt in SUBSTITUTIONS:
        colors.extend([SUBSTITUTION_COLORS[ref, alt]] * 16)
    x = np.arange(96)
    ax.bar(x, signature, color=colors, edgecolor='none', width=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([ch.split('[')[0] + ch.split(']')[1] for ch in SBS96_CHANNELS], rotation=90, fontsize=6, family='monospace')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i, (ref, alt) in enumerate(SUBSTITUTIONS):
        ax.axvspan(i * 16 - 0.5, (i + 1) * 16 - 0.5, ymin=0.97, ymax=1.0, color=SUBSTITUTION_COLORS[ref, alt], clip_on=False)
        ax.text(i * 16 + 7.5, signature.max() * 1.05, f'{ref}>{alt}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    return ax

def plot_signature_panel(W: np.ndarray, titles: list[str] | None=None):
    K = W.shape[1]
    fig, axes = plt.subplots(K, 1, figsize=(12, 2.8 * K), squeeze=False)
    axes = axes.ravel()
    for k in range(K):
        t = titles[k] if titles else f'Signature {k}'
        plot_signature(W[:, k], title=t, ax=axes[k])
    fig.tight_layout()
    return fig
