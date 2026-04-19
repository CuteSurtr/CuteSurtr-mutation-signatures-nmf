from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
EPS = 1e-10

@dataclass
class NMF:
    n_components: int
    loss: str = 'kl'
    max_iter: int = 500
    tol: float = 0.0001
    init: str = 'nndsvd'
    random_state: int | None = 0
    loss_history_: list[float] = field(default_factory=list, init=False)
    n_iter_: int = field(default=0, init=False)
    W_: np.ndarray | None = field(default=None, init=False)
    H_: np.ndarray | None = field(default=None, init=False)

    def fit_transform(self, V: np.ndarray) -> np.ndarray:
        V = np.asarray(V, dtype=np.float64)
        if (V < 0).any():
            raise ValueError('NMF requires a non-negative input matrix.')
        if self.loss not in {'frobenius', 'kl'}:
            raise ValueError(f'unknown loss: {self.loss!r}')
        W, H = self._initialize(V)
        self.loss_history_ = [self._compute_loss(V, W, H)]
        for it in range(self.max_iter):
            if self.loss == 'frobenius':
                H = self._frobenius_update_H(V, W, H)
                W = self._frobenius_update_W(V, W, H)
            else:
                H = self._kl_update_H(V, W, H)
                W = self._kl_update_W(V, W, H)
            loss = self._compute_loss(V, W, H)
            self.loss_history_.append(loss)
            prev = self.loss_history_[-2]
            if prev > 0 and abs(prev - loss) / prev < self.tol:
                self.n_iter_ = it + 1
                break
        else:
            self.n_iter_ = self.max_iter
        W, H = self._normalize(W, H)
        self.W_, self.H_ = (W, H)
        return W

    def fit(self, V: np.ndarray) -> 'NMF':
        self.fit_transform(V)
        return self

    @staticmethod
    def _frobenius_update_H(V, W, H):
        numer = W.T @ V
        denom = W.T @ W @ H + EPS
        return H * (numer / denom)

    @staticmethod
    def _frobenius_update_W(V, W, H):
        numer = V @ H.T
        denom = W @ H @ H.T + EPS
        return W * (numer / denom)

    @staticmethod
    def _kl_update_H(V, W, H):
        WH = W @ H + EPS
        numer = W.T @ (V / WH)
        denom = W.sum(axis=0, keepdims=True).T + EPS
        return H * (numer / denom)

    @staticmethod
    def _kl_update_W(V, W, H):
        WH = W @ H + EPS
        numer = V / WH @ H.T
        denom = H.sum(axis=1, keepdims=True).T + EPS
        return W * (numer / denom)

    def _compute_loss(self, V, W, H) -> float:
        WH = W @ H
        if self.loss == 'frobenius':
            return 0.5 * float(np.sum((V - WH) ** 2))
        mask = V > 0
        kl = np.zeros_like(V)
        kl[mask] = V[mask] * np.log(V[mask] / (WH[mask] + EPS))
        kl -= V
        kl += WH
        return float(kl.sum())

    def _initialize(self, V):
        F, N = V.shape
        K = self.n_components
        rng = np.random.default_rng(self.random_state)
        if self.init == 'random':
            scale = np.sqrt(V.mean() / K)
            W = rng.random((F, K)) * scale
            H = rng.random((K, N)) * scale
            return (W, H)
        if self.init == 'nndsvd':
            return self._nndsvd(V, K)
        raise ValueError(f'unknown init: {self.init!r}')

    @staticmethod
    def _nndsvd(V, K):
        U, S, Vt = np.linalg.svd(V, full_matrices=False)
        W = np.zeros((V.shape[0], K))
        H = np.zeros((K, V.shape[1]))
        W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
        H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])
        for j in range(1, K):
            u, v = (U[:, j], Vt[j, :])
            u_pos, u_neg = (np.maximum(u, 0), np.maximum(-u, 0))
            v_pos, v_neg = (np.maximum(v, 0), np.maximum(-v, 0))
            u_pos_n, u_neg_n = (np.linalg.norm(u_pos), np.linalg.norm(u_neg))
            v_pos_n, v_neg_n = (np.linalg.norm(v_pos), np.linalg.norm(v_neg))
            pos_term = u_pos_n * v_pos_n
            neg_term = u_neg_n * v_neg_n
            if pos_term >= neg_term:
                sigma = pos_term
                u_choice = u_pos / (u_pos_n + EPS)
                v_choice = v_pos / (v_pos_n + EPS)
            else:
                sigma = neg_term
                u_choice = u_neg / (u_neg_n + EPS)
                v_choice = v_neg / (v_neg_n + EPS)
            W[:, j] = np.sqrt(S[j] * sigma) * u_choice
            H[j, :] = np.sqrt(S[j] * sigma) * v_choice
        W[W < EPS] = EPS
        H[H < EPS] = EPS
        return (W, H)

    @staticmethod
    def _normalize(W, H):
        col_sums = W.sum(axis=0, keepdims=True) + EPS
        W = W / col_sums
        H = H * col_sums.T
        return (W, H)
