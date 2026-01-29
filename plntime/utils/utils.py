import torch
import torch.nn.functional as F


def cosine_similarity(A, B, eps=1e-8):
    """
    Compute the cosine similarity between two matrices.
    Parameters
    ----------
    A: torch.Tensor
    B: torch.Tensor
    eps: float

    Returns
    -------
    torch.Tensor
    """
    dot_product = torch.sum(A * B, dim=-1)
    norm_A = torch.norm(A, dim=-1)
    norm_B = torch.norm(B, dim=-1)
    return dot_product / (norm_A * norm_B + eps)


def viterbi_algorithm(
    alpha,
    tau,
    eps: float = 1e-32,
    return_indices: bool = False,
):
    """
    Batched Viterbi for n batches of d independent chains using:
      - initialization alpha = p(z_1 | x)
      - transitions tau_t = p(z_{t+1} | z_t, x)

    Inputs
    ------
    alpha : (n, T, d, C) or (n, d, C)
        If (n,T,d,C), only alpha[:,0] is used as initialization.
    tau : (n, T-1, d, C, C)
        Transition kernel per time.
    """
    with torch.no_grad():
        if tau.ndim != 5:
            raise ValueError(f"tau must have shape (n,T-1,d,C,C); got {tuple(tau.shape)}")

        n, Tm1, d, C, C2 = tau.shape
        if C != C2:
            raise ValueError(f"tau last two dims must be (C,C); got (C={C}, C2={C2})")
        T = Tm1 + 1

        # Extract initialization probability
        if alpha.ndim == 4:
            if alpha.shape != (n, T, d, C):
                raise ValueError(f"alpha shape mismatch: expected {(n,T,d,C)}, got {tuple(alpha.shape)}")
            alpha0 = alpha[:, 0]  # (n, d, C)
        elif alpha.ndim == 3:
            if alpha.shape != (n, d, C):
                raise ValueError(f"alpha shape mismatch: expected {(n,d,C)}, got {tuple(alpha.shape)}")
            alpha0 = alpha
        else:
            raise ValueError(f"alpha must have shape (n,T,d,C) or (n,d,C); got {tuple(alpha.shape)}")

        log_alpha0 = torch.log(alpha0.clamp_min(eps))
        log_tau = torch.log(tau.clamp_min(eps))

        device = tau.device

        # delta[b,k,c] = best log-score up to time t ending in state c
        delta = log_alpha0.clone()  # (n, d, C)
        # psi[b,t,k,c_next] = argmax c_prev
        psi = torch.empty((n, T, d, C), device=device, dtype=torch.long)
        psi[:, 0].zero_()  # unused at t=0

        for t in range(1, T):
            scores = delta.unsqueeze(-1) + log_tau[:, t - 1]   # (n, d, C_prev, C_next)
            best_prev, arg_prev = scores.max(dim=2)            # (n, d, C_next), (n, d, C_next)
            psi[:, t] = arg_prev
            delta = best_prev                                  # (n, d, C)

        # Last state selection
        last = delta.argmax(dim=-1)                            # (n, d)

        # Backtrack to find full path
        path_idx = torch.empty((n, T, d), device=device, dtype=torch.long)
        path_idx[:, T - 1] = last
        for t in range(T - 1, 0, -1):
            prev = torch.gather(psi[:, t], dim=2, index=path_idx[:, t].unsqueeze(-1)).squeeze(-1)
            path_idx[:, t - 1] = prev

        path_1hot = F.one_hot(path_idx, num_classes=C).to(dtype=tau.dtype)  # (n, T, d, C)

        if return_indices:
            return path_1hot, path_idx
        return path_1hot


