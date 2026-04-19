from __future__ import annotations
import numpy as np
import pytest
from sklearn.decomposition import NMF as SkNMF
from mutsig.cosmic import cosine_similarity_matrix
from mutsig.features import SBS96_CHANNELS, synthetic_mutation_matrix
from mutsig.nmf import NMF

def test_channels_are_96():
    assert len(SBS96_CHANNELS) == 96
    assert len(set(SBS96_CHANNELS)) == 96

@pytest.mark.parametrize('loss', ['frobenius', 'kl'])
def test_loss_monotone_decreasing(loss):
    V, _, _ = synthetic_mutation_matrix(n_samples=10, n_signatures=3, seed=1)
    model = NMF(n_components=3, loss=loss, max_iter=200, random_state=1)
    model.fit(V.values)
    diffs = np.diff(model.loss_history_)
    assert (diffs <= 1e-06).all(), f'loss increased at some iteration; max increase = {diffs.max():.3e}'

def test_outputs_nonnegative_and_normalized():
    V, _, _ = synthetic_mutation_matrix(n_samples=15, n_signatures=4, seed=2)
    model = NMF(n_components=4, loss='kl', max_iter=300, random_state=2)
    model.fit(V.values)
    assert (model.W_ >= 0).all()
    assert (model.H_ >= 0).all()
    np.testing.assert_allclose(model.W_.sum(axis=0), 1.0, atol=1e-06)

def test_recovers_synthetic_signatures():
    V, W_true, _ = synthetic_mutation_matrix(n_samples=80, n_signatures=4, avg_mutations=1500, seed=3)
    model = NMF(n_components=4, loss='kl', max_iter=2000, tol=1e-06, random_state=3)
    model.fit(V.values)
    sims = cosine_similarity_matrix(model.W_, W_true)
    best = sims.max(axis=1)
    assert (best > 0.9).all(), f'weakest recovery cosine = {best.min():.3f}'

def test_agrees_with_sklearn_on_small_matrix():
    V, _, _ = synthetic_mutation_matrix(n_samples=30, n_signatures=3, seed=4)
    ours = NMF(n_components=3, loss='frobenius', max_iter=1000, tol=1e-07, random_state=4)
    ours.fit(V.values)
    sk = SkNMF(n_components=3, init='nndsvda', beta_loss='frobenius', solver='mu', max_iter=1000, tol=1e-07, random_state=4)
    sk.fit(V.values.T)
    sk_signatures = sk.components_.T
    sims = cosine_similarity_matrix(ours.W_, sk_signatures)
    best = sims.max(axis=1)
    assert (best > 0.95).all(), f'disagreement with sklearn: {best}'
