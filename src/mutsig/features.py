from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd
BASES = ('A', 'C', 'G', 'T')
PYRIMIDINES = ('C', 'T')
SUBSTITUTIONS = (('C', 'A'), ('C', 'G'), ('C', 'T'), ('T', 'A'), ('T', 'C'), ('T', 'G'))
COMPLEMENT = str.maketrans('ACGT', 'TGCA')

def _revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def _build_channels() -> list[str]:
    channels = []
    for ref, alt in SUBSTITUTIONS:
        for up in BASES:
            for down in BASES:
                channels.append(f'{up}[{ref}>{alt}]{down}')
    return channels
SBS96_CHANNELS: list[str] = _build_channels()
_CHANNEL_INDEX = {ch: i for i, ch in enumerate(SBS96_CHANNELS)}

def classify_snv(ref_tri: str, alt: str) -> str:
    if len(ref_tri) != 3 or any((b not in BASES for b in ref_tri)):
        raise ValueError(f'bad trinucleotide: {ref_tri!r}')
    if alt not in BASES:
        raise ValueError(f'bad alt base: {alt!r}')
    ref = ref_tri[1]
    up, down = (ref_tri[0], ref_tri[2])
    if ref not in PYRIMIDINES:
        ref_tri = _revcomp(ref_tri)
        alt = alt.translate(COMPLEMENT)
        ref = ref_tri[1]
        up, down = (ref_tri[0], ref_tri[2])
    return f'{up}[{ref}>{alt}]{down}'

def extract_trinucleotide_from_gdc_context(context: str) -> str | None:
    if not isinstance(context, str) or len(context) != 11:
        return None
    tri = context[4:7].upper()
    if any((b not in 'ACGT' for b in tri)):
        return None
    return tri

def maf_to_sbs96(mutations: pd.DataFrame, sample_col: str='Tumor_Sample_Barcode', context_col: str='CONTEXT', ref_col: str='Reference_Allele', alt_col: str='Tumor_Seq_Allele2') -> pd.DataFrame:
    df = mutations.copy()
    df['ref_tri'] = df[context_col].map(extract_trinucleotide_from_gdc_context)
    df = df.dropna(subset=['ref_tri'])
    df = df[df['ref_tri'].str[1] == df[ref_col].str.upper()]
    return build_feature_matrix(df, sample_col=sample_col, context_col='ref_tri', alt_col=alt_col)

def build_feature_matrix(mutations: pd.DataFrame, sample_col: str='sample', context_col: str='ref_tri', alt_col: str='alt') -> pd.DataFrame:
    required = {sample_col, context_col, alt_col}
    missing = required - set(mutations.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    channels = mutations.apply(lambda r: classify_snv(r[context_col], r[alt_col]), axis=1)
    counts = pd.DataFrame({'sample': mutations[sample_col], 'channel': channels}).groupby(['channel', 'sample']).size().unstack(fill_value=0)
    V = counts.reindex(SBS96_CHANNELS, fill_value=0)
    return V.astype(np.float64)

def synthetic_mutation_matrix(n_samples: int=20, n_signatures: int=3, avg_mutations: int=300, seed: int=0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W_true = rng.dirichlet(np.full(96, 0.5), size=n_signatures).T
    H_true = rng.gamma(2.0, avg_mutations / n_signatures / 2.0, size=(n_signatures, n_samples))
    mean = W_true @ H_true
    V = rng.poisson(mean).astype(np.float64)
    df = pd.DataFrame(V, index=SBS96_CHANNELS, columns=[f'S{i}' for i in range(n_samples)])
    return (df, W_true, H_true)

def iter_channels() -> Iterable[str]:
    return iter(SBS96_CHANNELS)
