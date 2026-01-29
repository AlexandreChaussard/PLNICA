import math
import torch
import torch.nn as nn


class CountsPreprocessing(nn.Module):

    def __init__(self, preprocessing):
        """
        Preprocessing of count data module. If no preprocessing is provided, returns the same as the input.
        Parameters
        ----------
        preprocessing: str or list[str]
        """
        super(CountsPreprocessing, self).__init__()
        if preprocessing is None:
            preprocessing = []
        self.log_transform = 'log' in preprocessing
        self.proportion = 'proportion' in preprocessing

    def forward(self, X):
        x = X.clone()
        if self.proportion:
            x = x / (torch.sum(x, dim=-1, keepdim=True) + 1e-8)
        if self.log_transform is True and not self.proportion:
            x = torch.log(X + 0.5)
        return x


class Amortizer(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, after_network, dropout=0.0):
        """
        Amortizer module with GRU and after network.
        Parameters
        ----------
        input_size: int
        hidden_size: int
        num_layers: int
        after_network: nn.Module
        dropout: float
        """
        super(Amortizer, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.network = after_network

    def forward(self, x, lengths=None):
        out, _ = self.gru(x)  # out: (batch, T, hidden)

        if lengths is None:
            return self.network(out[:, -1, :])
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        return self.network(out.gather(1, idx).squeeze(1))


class Scale(nn.Module):
    def __init__(self, rho: float):
        """
        Scale module that multiplies input by a constant factor rho.
        Parameters
        ----------
        rho: float
        """
        super().__init__()
        self.rho = rho
    def forward(self, x):
        return self.rho * x


class BoundedVariance(nn.Module):
    def __init__(self, min_var=1e-3, max_var=10.):
        """
        Module that maps input to a variance bounded between min_var and max_var.
        Parameters
        ----------
        min_var: float
        """
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def forward(self, x):
        log_min, log_max = math.log(self.min_var), math.log(self.max_var)
        log_psi = log_min + (log_max - log_min) * torch.sigmoid(x)
        return torch.exp(log_psi)


class GammaICA(nn.Module):
    def __init__(self, K, d, eps=1e-8, device='cpu', dtype=torch.float64):
        """
        Linear ICA basis module.
        Parameterizes the mixing map with a learnable weight matrix W of shape (K, d).
        The columns of W are normalized to have unit L2 norm to ensure identifiability.
        Parameters
        ----------
        K: int
        d: int
        eps: float
        device: str
        dtype: torch.dtype
        """
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(0.01 * torch.randn(K, d, device=device, dtype=dtype))

    def normalize(self):
        col_norm = torch.linalg.norm(self.W, dim=0, keepdim=True) + self.eps
        base = self.W / col_norm
        return base