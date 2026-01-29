import numpy as np

from picard import picard
from coroica.uwedgeica import UwedgeICA


class BatchedPicardICA:

    def __init__(self, d: int, random_state: int | None = None):
        self.d = int(d)
        self.random_state = random_state

        self.V_ = None      # (d, K)
        self.A_ = None      # (K, d)
        self.mean_ = None   # (K,)
        self.n_iter_ = None
        self.K_ = None      # Picard prewhitening matrix (d, K)
        self.W_ = None      # Picard unmixing in whitened space (d, d)

    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"X must have shape (n_trials, T, K). Got {X.shape}")
        n, T, K = X.shape
        if T < 2:
            raise ValueError("Need T >= 2.")
        return X, n, T, K

    def fit(self, X):
        X, n, T, K = self._check_X(X)
        if self.d > K:
            raise ValueError(f"d={self.d} cannot exceed K={K}.")

        Xflat = X.reshape(n * T, K).T  # (K, n*T)

        Kwhiten, W, _Y, X_mean, n_iter = picard(
            Xflat,
            n_components=self.d,
            return_X_mean=True,
            return_n_iter=True,
            random_state=self.random_state,
        )

        self.K_ = Kwhiten        # (d, K) since whiten=True
        self.W_ = W              # (d, d)
        self.mean_ = X_mean      # (K,)
        self.n_iter_ = int(n_iter)

        self.V_ = self.W_ @ self.K_  # (d, K)

        VVt = self.V_ @ self.V_.T
        self.A_ = self.V_.T @ np.linalg.inv(VVt)  # (K, d)

        return self

    def transform(self, X):
        if self.V_ is None or self.mean_ is None:
            raise RuntimeError("Call fit(X) before transform(X).")

        X, n, T, K = self._check_X(X)
        if K != self.V_.shape[1]:
            raise ValueError(f"Feature dim mismatch: fit on K={self.V_.shape[1]} but got K={K}.")

        Xflat = X.reshape(n * T, K)                 # (n*T, K)
        Xflat = Xflat - self.mean_[None, :]         
        Sflat = Xflat @ self.V_.T                   # (n*T, d)
        return Sflat.reshape(n, T, self.d)

    def inverse_transform(self, S):
        if self.A_ is None or self.mean_ is None:
            raise RuntimeError("Call fit(X) before inverse_transform(S).")

        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 3:
            raise ValueError(f"S must have shape (n_trials, T, d). Got {S.shape}")
        n, T, d = S.shape
        if d != self.d:
            raise ValueError(f"Last dim must be d={self.d}. Got {d}")

        Sflat = S.reshape(n * T, d)                 # (n*T, d)
        Xc_hat = Sflat @ self.A_.T                  # (n*T, K)
        X_hat = Xc_hat + self.mean_[None, :]        # (n*T, K)
        return X_hat.reshape(n, T, -1)

    def get_unmixing(self):
        if self.V_ is None:
            raise RuntimeError("Call fit(X) first.")
        return self.V_

    def get_mixing(self):
        if self.A_ is None:
            raise RuntimeError("Call fit(X) first.")
        return self.A_


class BatchedUwedgeICA:
    def __init__(self, d, window=10, step=10, lags=None,
                 center="global", reg=1e-3, instantcov=True,
                 max_iter=1000, tol=1e-12, minimize_loss=True,
                 condition_threshold=1e9):
        self.d = int(d)
        self.window = int(window)
        self.step = int(step)
        self.center = center
        self.reg = float(reg)
        self.instantcov = bool(instantcov)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.minimize_loss = bool(minimize_loss)
        self.condition_threshold = condition_threshold
        self.lags = None if lags is None else list(map(int, lags))

        self.mean_ = None
        self.W_white_ = None
        self.model_ = None
        self.V_ = None
        self.A_ = None

    def _center(self, X):
        if self.center == "global":
            mu = X.reshape(-1, X.shape[-1]).mean(axis=0)
            return X - mu[None, None, :], mu
        if self.center == "per_trial":
            mu_trial = X.mean(axis=1, keepdims=True)
            return X - mu_trial, mu_trial.mean(axis=0).squeeze(0)
        raise ValueError("center must be 'global' or 'per_trial'.")

    def _center_transform(self, X):
        if self.center == "global":
            return X - self.mean_[None, None, :]
        return X - X.mean(axis=1, keepdims=True)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, T, K = X.shape

        if self.lags is None:
            self.lags = list(range(1, min(self.window, 6)))

        if self.step != self.window:
            raise ValueError("Use step == window (non-overlapping windows).")

        T_eff = (T // self.window) * self.window
        X = X[:, :T_eff, :]

        Xc, mu = self._center(X)
        self.mean_ = mu
        Xflat = Xc.reshape(n * T_eff, K)

        # ridge-PCA whitening to d
        C0 = np.cov(Xflat, rowvar=False) + self.reg * np.eye(K)
        evals, evecs = np.linalg.eigh(C0)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order][:self.d], self.reg)
        evecs = evecs[:, order][:, :self.d]
        self.W_white_ = (np.diag(1.0 / np.sqrt(evals)) @ evecs.T)  # (d, K)
        Y = Xflat @ self.W_white_.T  # (n*T_eff, d)

        n_blocks = T_eff // self.window
        block_id = (np.arange(T_eff) // self.window)
        partition_index = (np.arange(n)[:, None] * n_blocks + block_id[None, :]).reshape(-1).astype(np.int64)

        model = UwedgeICA(
            n_components=self.d,
            n_components_uwedge=self.d,
            rank_components=True, 
            timelags=[tau for tau in self.lags if tau < self.window],
            instantcov=self.instantcov,
            max_iter=self.max_iter,
            tol=self.tol,
            minimize_loss=self.minimize_loss,
            condition_threshold=self.condition_threshold,
        )
        model.fit(Y, partition_index=partition_index)
        self.model_ = model

        self.V_ = model.V_ @ self.W_white_          # (d, K)
        self.A_ = np.linalg.pinv(self.V_)          # (K, d)
        return self

    def transform(self, X):
        if self.V_ is None or self.mean_ is None:
            raise RuntimeError("Call fit(X) before transform(X).")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"X must have shape (n_trials, T, K). Got {X.shape}")
        n, T, K = X.shape
        if self.V_.shape[1] != K:
            raise ValueError(f"Feature dim mismatch: fit on K={self.V_.shape[1]} but got K={K}.")

        Xc = self._center_transform(X)
        Sflat = Xc.reshape(n * T, K) @ self.V_.T     # (n*T, d)
        return Sflat.reshape(n, T, self.d)

    def inverse_transform(self, S):
        if self.A_ is None or self.mean_ is None:
            raise RuntimeError("Call fit(X) before inverse_transform(S).")

        S = np.asarray(S, dtype=np.float64)
        if S.ndim != 3:
            raise ValueError(f"S must have shape (n_trials, T, d). Got {S.shape}")
        n, T, d = S.shape
        if d != self.d:
            raise ValueError(f"Last dim must be d={self.d}. Got {d}")

        Xflat = S.reshape(n * T, d) @ self.A_.T      # (n*T, K)
        Xhat = Xflat.reshape(n, T, -1)

        if self.center == "global":
            Xhat += self.mean_[None, None, :]

        return Xhat

    def get_unmixing(self):
        return self.V_

    def get_mixing(self):
        return self.A_