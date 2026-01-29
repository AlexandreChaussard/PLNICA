import numpy as np


class VAR1:
    """
    Time-varying VAR(1) on a panel of trajectories.
    """

    def __init__(self, jitter: float = 1e-9, random_state=None, ridge: float = 1e-3):
        self.jitter = float(jitter)
        self.rng = np.random.default_rng(random_state)
        self.ridge = float(ridge)

        self.mu1_ = None            # (d,)
        self.Sigma1_ = None         # (d,d)
        self.A_ = None              # (T-1,d,d)
        self.b_ = None              # (T-1,d)
        self.Sigma_ = None          # (T-1,d,d)

        self.T_ = None
        self.d_ = None
        self.is_fitted_ = False

    @staticmethod
    def _symmetrize(M):
        return 0.5 * (M + M.T)

    def _make_spd(self, M):
        """
        Make a symmetric positive definite matrix by eigenvalue clipping + jitter.
        """
        M = self._symmetrize(M)
        vals, vecs = np.linalg.eigh(M)
        floor = self.jitter
        vals = np.maximum(vals, floor)
        return (vecs * vals) @ vecs.T

    def fit(self, x: np.ndarray):
        x = np.asarray(x)
        if x.ndim != 3:
            raise ValueError(f"x must have shape (n, T, d). Got ndim={x.ndim}.")
        n, T, d = x.shape
        if T < 2:
            raise ValueError("Need T >= 2 to fit transitions.")

        self.T_ = int(T)
        self.d_ = int(d)

        # Initial distribution MLE: X_1 ~ N(mu1, Sigma1)
        X1 = x[:, 0, :]  # (n,d)
        mu1 = X1.mean(axis=0)
        E1 = X1 - mu1
        Sigma1 = (E1.T @ E1) / n  # MLE (not unbiased)
        Sigma1 = self._make_spd(Sigma1)

        # Time-varying transitions: fit each t separately using the n paired samples
        A = np.zeros((T - 1, d, d), dtype=float)
        b = np.zeros((T - 1, d), dtype=float)
        Sigma = np.zeros((T - 1, d, d), dtype=float)

        ones = np.ones((n, 1), dtype=float)

        for t in range(T - 1):
            Xt = x[:, t, :]        # (n,d)
            Xtp1 = x[:, t + 1, :]  # (n,d)

            Z = np.concatenate([ones, Xt], axis=1)  # (n, d+1)
            Y = Xtp1                                # (n, d)

            ZTZ = Z.T @ Z
            penalty = self.ridge * np.eye(d + 1, dtype=float)
            penalty[0, 0] = 0.0
            B_hat = np.linalg.solve(ZTZ + penalty, Z.T @ Y)  # (d+1, d)

            b_t = B_hat[0]          # (d,)
            A_t = B_hat[1:].T       # (d,d)

            R = Y - Z @ B_hat       
            Sigma_t = (R.T @ R) / n 
            Sigma_t = self._make_spd(Sigma_t)

            A[t] = A_t
            b[t] = b_t
            Sigma[t] = Sigma_t

        self.mu1_ = mu1
        self.Sigma1_ = Sigma1
        self.A_ = A
        self.b_ = b
        self.Sigma_ = Sigma

        self.is_fitted_ = True
        return self

    def sample(self, n: int, T: int | None = None, random_state=None):
        rng = self.rng if random_state is None else np.random.default_rng(random_state)

        T_req = self.T_ if T is None else int(T)
        if T_req < 1:
            raise ValueError("T must be >= 1.")
        if T_req > self.T_:
            raise ValueError(
                f"Requested T={T_req} exceeds fitted horizon T_={self.T_} "
                "for time-varying parameters."
            )

        d = self.d_
        out = np.zeros((n, T_req, d), dtype=float)

        # Sample X_1
        out[:, 0, :] = rng.multivariate_normal(self.mu1_, self.Sigma1_, size=n)

        # Sample transitions
        for t in range(T_req - 1):
            mean = out[:, t, :] @ self.A_[t].T + self.b_[t]  # (n,d)
            eps = rng.multivariate_normal(np.zeros(d), self.Sigma_[t], size=n)
            out[:, t + 1, :] = mean + eps

        return out

    def predict(self, x_hist: np.ndarray):
        x_hist = np.asarray(x_hist)
        if x_hist.ndim != 3:
            raise ValueError("x_hist must have shape (n, t0, d).")

        n, t0, d = x_hist.shape
        if d != self.d_:
            raise ValueError(f"Dimension mismatch: x_hist has d={d}, model has d_={self.d_}.")
        if t0 < 1:
            raise ValueError("Need at least one observation in x_hist (t0 >= 1).")
        if t0 > self.T_:
            raise ValueError(f"t0={t0} cannot exceed fitted horizon T_={self.T_}.")

        T = self.T_
        out = np.zeros((n, T, d), dtype=float)
        out[:, :t0, :] = x_hist

        # Complete deterministically using conditional mean
        for t in range(t0 - 1, T - 1):
            out[:, t + 1, :] = out[:, t, :] @ self.A_[t].T + self.b_[t]

        return out
