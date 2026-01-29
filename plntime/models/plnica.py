import random
import itertools
import torch
import torch.nn as nn
import tqdm
from torch import log, exp

from plntime.utils.modules import Amortizer, CountsPreprocessing, Scale, BoundedVariance, GammaICA
from plntime.utils.utils import viterbi_algorithm

class PLNICA(nn.Module):

    def __init__(
            self,
            counts,
            times=None,
            covariates=None,
            latent_size=5,
            n_dynamics=1,
            network_params=None,
            predictive=False,
            mean_field=False,
            offsets='zero',
            regularization=0.,
            dropout=0.,
            bias=True,
            device='cpu',
            precision=torch.float64,
            seed=None
    ):
        """
        Poisson Log-Normal Independent Component Analysis with switching linear dynamics in the latent space (ARPLN-ICA).

        Example usage:
        >>> from plntime.import PLNICA
        >>> # counts: torch.Tensor of shape (n, T, K)
        >>> model = PLNICA(counts, latent_size=5, n_dynamics=2, device='cuda')
        >>> losses = model.fit(n_epochs=500, learning_rate=1e-3)

        Parameters
        ----------
        counts: torch.Tensor
            Tensor of shape (n, T, K) with count observations.
        times: torch.Tensor or None
            Tensor of shape (n, T) with observation times. If None, assumes regular sampling.
        covariates: torch.Tensor or None
            Tensor of shape (n, C) with covariates. If None, only intercept is used.
        latent_size: int
            Dimensionality of the latent space.
        n_dynamics: int
            Number of switching dynamics states.
        network_params: dict or None
            Dictionary of network parameters. If None, default parameters are used.
            Parameters can be provided for:
            - amortized_size: int (hidden size of the amortizer network); default: 32
            - n_layers_amortizer: int (number of layers in the amortizer network); default: 2
            - n_output_layers_amortizer: int (number of output layers in the amortizer network); default: 1
            - shared_network_size: list of int (hidden sizes of the shared network); default: [16, 16]
            - n_output_layers_shared: int (number of output layers in the shared network); default: 2
        predictive: bool
            If True, uses filtering parameterization (compatible with forecasting); if False, uses smoothing.
        mean_field: bool
            If True, uses mean-field approximation in the latent dynamics.
        offsets: str or torch.Tensor
            'zero', 'logsum', or tensor of shape (n, T) with offsets for the Poisson rates.
        regularization: float
            L2 regularization strength on the columns of the mixing matrix Gamma.
        dropout: float
            Dropout rate in the neural networks.
        bias: bool
            If True, includes bias terms for covariate effects.
        device: str
            'cpu' or 'cuda' for computation device.
        precision: torch.dtype
            torch.float32 or torch.float64 for computation precision.
        seed: int or None
            Random seed for reproducibility.
        """
        super().__init__()

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print('Warning: CUDA device requested but not available. Using CPU instead.')
        self.device = device
        self.predictive = predictive
        self.mean_field = mean_field
        self.bias = bias
        self.seed = seed
        self.regularization = regularization
        if self.seed is not None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)

        self.counts = counts.clone().to(self.device, dtype=precision)
        if offsets == 'zero':
            self.offsets = torch.zeros((self.counts.shape[0], self.counts.shape[1]), device=self.device, dtype=precision)
        elif offsets == 'logsum':
            self.offsets = torch.log(self.counts.sum(dim=-1) + 1.).to(self.device, dtype=precision)
        else:
            try:
                self.offsets = offsets.clone().to(self.device, dtype=precision)
            except AttributeError:
                raise ValueError("Offsets must be 'zero', 'logsum', or a torch.Tensor of shape (n,T).")
        if times is None:
            # If no times provided, assume regular sampling
            self.times = torch.arange(
                self.counts.shape[1],
                device=self.device,
                dtype=precision).unsqueeze(0).expand(self.counts.shape[0], -1)  # (n, T)
        else:
            # Assert that the provided times have the correct shape
            assert times.shape == (self.counts.shape[0], self.counts.shape[1]), "Times tensor must have shape (n, T)."
            self.times = times.clone().to(self.device, dtype=precision)
        if covariates is not None:
            assert covariates.shape[0] == self.counts.shape[0], "Covariates tensor must have shape (n, C)."
            self.covariates = torch.tensor(covariates).to(self.device, dtype=precision)
        else:
            # Add intercept covariate by default
            self.covariates = torch.ones(self.counts.shape[0], 1, device=self.device, dtype=precision)
        self.cov_size = self.covariates.shape[1]
        self.n = self.counts.shape[0]
        self.T = self.counts.shape[1]
        self.d = latent_size
        self.K = self.counts.shape[2]
        self.C = n_dynamics

        if network_params is None:
            network_params = {}
        # Amortizer network parameters
        if 'amortized_size' not in network_params:
            network_params['amortized_size'] = 32
        if 'n_layers_amortizer' not in  network_params:
            network_params['n_layers_amortizer'] = 2
        if 'n_output_layers_amortizer' not in network_params:
            network_params['n_output_layers_amortizer'] = 1
        # Shared network parameters for variational parameters
        if 'shared_network_size' not in network_params:
            network_params['shared_network_size'] = [16, 16]
        if 'n_output_layers_shared' not in network_params:
            network_params['n_output_layers_shared'] = 2
        self.network_params = network_params

        # Count preprocessing module used across all inputs
        self.count_preprocessing = CountsPreprocessing('proportion')

        # Amortizer of the counts chain in time
        amortized_size = network_params['amortized_size']
        n_layers_amortizer = network_params['n_layers_amortizer']
        n_output_layers_amortizer = network_params['n_output_layers_amortizer']
        self.amortizer_nn = nn.Sequential()
        for _ in range(n_output_layers_amortizer):
            self.amortizer_nn.add_module('amortizer_relu_' + str(_), nn.ReLU())
            self.amortizer_nn.add_module('amortizer_fc_' + str(_), nn.Linear(amortized_size, amortized_size))
        self.amortizer = Amortizer(
            input_size=self.K + 1 + self.cov_size,
            hidden_size=amortized_size,
            num_layers=n_layers_amortizer,
            after_network=self.amortizer_nn,
            dropout=dropout
        )

        # ------------------------------------
        # Variational network components
        # ------------------------------------

        # Input size: amortized input + counts + offset
        input_size = amortized_size + self.K + self.cov_size + 1

        # Shared network of the variational parameters
        self.shared_network = nn.Sequential()
        hidden_dims = [input_size] + network_params['shared_network_size']
        for i in range(len(hidden_dims) - 2):
            self.shared_network.add_module('dropout_' + str(i), nn.Dropout(p=dropout))
            self.shared_network.add_module('shared_fc_' + str(i), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.shared_network.add_module('shared_relu_' + str(i), nn.ReLU())
        self.shared_network.add_module('shared_out', nn.Linear(hidden_dims[-2], hidden_dims[-1]))

        # Forward means and biases
        self.m = nn.Sequential()
        self.psi_tilde = nn.Sequential()
        self.B_tilde = nn.Sequential()
        self.b_tilde = nn.Sequential()
        hidden_dims = [network_params['shared_network_size'][-1]]*network_params['n_output_layers_shared'] + [self.d]
        for i in range(len(hidden_dims) - 2):
            self.m.add_module('m_fc_' + str(i), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.m.add_module('m_relu_' + str(i), nn.ReLU())

            self.psi_tilde.add_module('psi_tilde_fc_' + str(i), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.psi_tilde.add_module('psi_tilde_relu_' + str(i), nn.ReLU())

            self.B_tilde.add_module('B_tilde_fc_' + str(i), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.B_tilde.add_module('B_tilde_relu_' + str(i), nn.ReLU())
            self.b_tilde.add_module('b_tilde_fc_' + str(i), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.b_tilde.add_module('b_tilde_relu_' + str(i), nn.ReLU())

        self.m.add_module('m_out', nn.Linear(hidden_dims[-2], hidden_dims[-1]))

        self.psi_tilde.add_module('psi_tilde_out', nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        self.psi_tilde.add_module('positive_output', BoundedVariance(1e-8, 30.))

        self.B_tilde.add_module('B_tilde_out', nn.Linear(hidden_dims[-2], hidden_dims[-1]))
        self.B_tilde.add_module('stabilize', nn.Tanh())
        self.B_tilde.add_module('scale_output', Scale(0.95))
        self.b_tilde.add_module('b_tilde_out', nn.Linear(hidden_dims[-2], hidden_dims[-1]))

        # ------------------------------------
        # Prior dynamic parameters
        # ------------------------------------
        self.pi = (torch.ones(self.d, self.C) / self.C).to(device=device, dtype=self.counts.dtype)             # Initial state probabilities (d, C)
        self.A = (torch.ones(self.d, self.C, self.C) / self.C).to(device=device, dtype=self.counts.dtype)      # Transition matrices (d, C, C)
        self.bar_b = (torch.randn(self.d, self.C)*.1).to(device=device, dtype=self.counts.dtype)               # Initial state mean (d, C)
        self.bar_psi = (torch.abs(torch.randn(self.d, self.C))*.1).to(device=device, dtype=self.counts.dtype)  # Initial state variance (d, C)
        self.B = (torch.randn(self.d, self.C)*.1).to(device=device, dtype=self.counts.dtype)                   # State transition coefficient (d, C)
        self.b = (torch.randn(self.d, self.C)*.1).to(device=device, dtype=self.counts.dtype)                   # State transition bias (d, C)
        self.psi = (torch.abs(torch.randn(self.d, self.C))*.1).to(device=device, dtype=self.counts.dtype)      # State variance (d, C)

        # ------------------------------------
        # Emission distribution parameters
        # ------------------------------------
        self.Gamma = GammaICA(K=self.K, d=self.d, eps=1e-6, device=device, dtype=self.counts.dtype)              # Mixing matrix (K, d)
        self.M = nn.Parameter(torch.randn(self.K, self.cov_size, device=device, dtype=self.counts.dtype) * 0.1)  # Covariate effects on the counts (K, d_covariates)
        if not self.bias:
            self.M.data = self.M.data.detach() * 0.

        # ------------------------------------
        # Identifiability parameters
        # ------------------------------------
        self.D = torch.eye(self.d, self.d, device=device, dtype=self.counts.dtype)  # Scaling matrix (d, d)
        self.P = torch.eye(self.d, self.d, device=device, dtype=self.counts.dtype)  # Permutation matrix (d, d)

        self.to(device=self.device, dtype=self.counts.dtype)


    def fit(self, n_epochs=400, tolerance=1e-4, wait=10, batch_size=64,
            learning_rate=1e-3, weight_decay=1e-4, gradient_clip=5., T_max=1., verbose=True):
        """
        Fit the ARPLN-ICA model via stochastic VEM with amortized inference.
        Parameters
        ----------
        n_epochs: int
            Number of training epochs.
        tolerance: float
            Tolerance for early stopping based on relative ELBO improvement.
        wait: int
            Number of epochs to wait for improvement before stopping.
        batch_size: int
            Mini-batch size for stochastic optimization.
        learning_rate: float
            Learning rate for the Adam optimizer.
        weight_decay: float
            Weight decay (L2 regularization) for the Adam optimizer.
        gradient_clip: float
            Maximum gradient norm for clipping.
        T_max: float
            Proportion of epochs for the learning rate scheduler.
        verbose: bool
            If True, displays a progress bar during training.

        Returns
        -------
        self
        """
        self.to(self.device)
        self.train()

        # Adam for everything except Gamma
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        # Scheduler to stabilize training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(T_max*n_epochs))

        dataset = torch.utils.data.TensorDataset(self.counts, self.offsets, self.covariates, self.times)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        losses = torch.zeros(n_epochs)
        wait_index = 0
        loop = tqdm.tqdm(range(n_epochs), leave=verbose,
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if verbose else range(n_epochs)

        for epoch in loop:
            epoch_loss = 0.0
            for (X_batch, O_batch, Cov_batch, times_batch) in dataloader:
                optimizer.zero_grad()
                # E-step
                latent_params = self.forward(X_batch, O_batch, Cov_batch, times_batch)
                expectations = self.E_step(latent_params)
                prior_params = {
                    'init_probs': self.pi,
                    'transition_probs': self.A,
                    'init_means': self.bar_b,
                    'init_covariance': self.bar_psi,
                    'forward_means': self.B,
                    'bias_forward_means': self.b,
                    'covariances': self.psi,
                    'Gamma': self.Gamma.normalize(),
                }
                # M-step on variational parameters via gradient ascent on the ELBO
                loss = -self.elbo(
                    X_batch, O_batch,
                    prior_params, latent_params, expectations
                )
                # Adding penalization on Gamma columns to promote sparsity
                loss += self.regularization * torch.sum(torch.linalg.norm(self.Gamma.W, dim=0, ord=2))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip)
                optimizer.step()
                if not self.bias:
                    self.M.data = self.M.data.detach() * 0.
                # M-step on the model parameters with closed-form updates
                # Detach for M-step
                latent_params_det = {k: v.detach() for k, v in latent_params.items()}
                expectations_det = {k: v.detach() for k, v in expectations.items()}
                self.M_step(X_batch, latent_params_det, expectations_det)


                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses[epoch] = avg_loss
            if epoch > 1:
                tol = abs(losses[epoch - 1] - losses[epoch - 2]) / abs(losses[epoch - 2])
            else:
                tol = float('inf')
            if verbose:
                loop.set_postfix({"Device": X_batch.device, "Loss": avg_loss})

            if epoch > 1 and tol < tolerance:
                wait_index += 1
                if wait_index == wait:
                    if verbose:
                        loop.set_postfix({"Loss": avg_loss, "Status": "Converged"})
                    break
            else:
                wait_index = 0

            scheduler.step()

        # Set small values in Gamma to zero after regularized training
        with torch.no_grad():
            self.Gamma.W.data[self.Gamma.W.data.abs() < 1e-8] = 0.0

        self.eval()
        return losses[:epoch + 1]

    def forward(self, X, O, Cov, times):
        latent_params = {}
        latent_params['forward_means'] = []
        latent_params['bias_forward_means'] = []
        latent_params['covariances'] = []
        latent_params['transition_probs'] = []
        batch_size = X.shape[0]
        T = X.shape[1]

        mask = times.clone()  # (batch_size, T)
        mask[torch.isfinite(times)] = 1.0
        mask[~torch.isfinite(times)] = 0.0
        lengths = mask.sum(dim=1).long()  # (batch_size,)

        for t in range(T):
            X_t = self.count_preprocessing(X[:, t, :])
            O_t = O[:, t].unsqueeze(-1)
            if self.predictive:
                if t > 0:
                    # Amortize the past observations X_{1:t} necessary for filtering parameterization
                    X_cat = torch.cat([
                        self.count_preprocessing(X[:, :t+1, :]),
                        O[:, :t+1].unsqueeze(-1),
                        Cov.unsqueeze(1).expand(-1, t+1, -1)
                    ],
                        dim=-1
                    )  # (batch_size, t+1, counts_dim + cov_dim + 1))
                    X_amortized = self.amortizer(X_cat, lengths.clamp_max(t+1))  # (batch_size, amortized_size)
                else:
                    # Padding for t=0
                    X_amortized = torch.zeros(
                        batch_size,
                        self.network_params['amortized_size']
                    ).to(X.device, dtype=X.dtype)    # (batch_size, amortized_size)
            else:
                # Amortize the full observations X_{1:T} for smoothing parameterization
                X_cat = torch.cat(
                    [
                        self.count_preprocessing(X),
                        O.unsqueeze(-1),
                        Cov.unsqueeze(1).expand(-1, T, -1)
                    ], dim=-1)            # (batch_size, t+1, counts_dim + cov_dim 1))
                X_amortized = self.amortizer(X_cat, lengths)     # (batch_size, amortized_size)
            X_cat = torch.cat([X_amortized, X_t, O_t, Cov], dim=1)
            # Preprocess [X_t, X_amortized, O_t] through shared network
            x = self.shared_network(X_cat)
            if t == 0:
                # Initial variational latent state mean
                m = self.m(x)                           # (batch_size, latent_dim)
                latent_params['init_means'] = m
                # Initial variational latent state diagonal variance terms
                psi_tilde = self.psi_tilde(x)           # (batch_size, latent_dim)
                latent_params['covariances'] = [psi_tilde]

                # Initial switching state probabilities
                sigma = psi_tilde.unsqueeze(-1).expand(-1, -1, self.C)            # (batch_size, latent_dim, switch_dim)
                mu = m.unsqueeze(-1).expand(-1, -1, self.C)                       # (batch_size, latent_dim, switch_dim)
                bar_b = self.bar_b.unsqueeze(0).expand(batch_size, -1, -1)        # (batch_size, latent_dim, switch_dim)
                bar_psi = self.bar_psi.unsqueeze(0).expand(batch_size, -1, -1)    # (batch_size, latent_dim, switch_dim)
                pi = self.pi.unsqueeze(0).expand(batch_size, -1, -1)              # (batch_size, latent_dim, switch_dim)

                log_e_1 = - (sigma + (mu - bar_b)**2) / (2*bar_psi) - log(bar_psi) / 2 # (batch_size, latent_dim, switch_dim)
                log_nu = log(pi) + log_e_1                                             # (batch_size, latent_dim, switch_dim)
                log_nu = log_nu - torch.logsumexp(log_nu, dim=-1, keepdim=True)        # (batch_size, latent_dim, switch_dim)
                nu = torch.clamp(exp(log_nu), min=1e-8) # Stabilize computations
                nu = nu / nu.sum(dim=-1, keepdim=True)  # Renormalize after stabilizing
                latent_params['init_probs'] = nu
            else:
                b_tilde = self.b_tilde(x)         # (batch_size, latent_dim)
                # If mean-field, the dependency on the previous latent state is removed
                if self.mean_field:
                    B_tilde = torch.zeros_like(b_tilde)     # (batch_size, latent_dim)
                else:
                    B_tilde = self.B_tilde(x)               # (batch_size, latent_dim)
                latent_params['forward_means'] += [B_tilde]
                latent_params['bias_forward_means'] += [b_tilde]
                # Transition variational latent state diagonal variance terms
                psi_tilde = self.psi_tilde(x)     # (batch_size, latent_dim)
                latent_params['covariances'] += [psi_tilde]

                # Transition switching state probabilities
                B_tilde_star = B_tilde.unsqueeze(-1).expand(-1, -1, self.C)      # (batch_size, latent_dim, switch_dim)
                b_tilde_star = b_tilde.unsqueeze(-1).expand(-1, -1, self.C)      # (batch_size, latent_dim, switch_dim)
                psi_tilde_star = psi_tilde.unsqueeze(-1).expand(-1, -1, self.C)  # (batch_size, latent_dim, switch_dim)
                mu_prev = mu
                mu = B_tilde_star * mu + b_tilde_star                            # (batch_size, latent_dim, switch_dim)
                sigma_prev = sigma
                sigma = B_tilde_star**2 * sigma + psi_tilde_star                 # (batch_size, latent_dim, switch_dim)
                cov = B_tilde_star * sigma_prev                                  # (batch_size, latent_dim, switch_dim)
                A = self.A.unsqueeze(0).expand(batch_size, -1, -1, -1)           # (batch_size, latent_dim, switch_dim, switch_dim)

                psi = self.psi.unsqueeze(0).expand(batch_size, -1, -1)           # (batch_size, latent_dim, switch_dim)
                B = self.B.unsqueeze(0).expand(batch_size, -1, -1)               # (batch_size, latent_dim, switch_dim)
                b = self.b.unsqueeze(0).expand(batch_size, -1, -1)               # (batch_size, latent_dim, switch_dim)

                log_e_t = - (sigma + B**2 * sigma_prev + (mu - B * mu_prev - b)**2 - 2 * B * cov)/(2*psi) - log(psi) / 2  # (batch_size, latent_dim, switch_dim)
                log_e_t = log_e_t.unsqueeze(-2).expand(-1, -1, self.C, -1) # (batch_size, latent_dim, switch_dim, switch_dim)
                log_tau = log(A) + log_e_t                                 # (batch_size, latent_dim, switch_dim, switch_dim)
                log_tau = log_tau - torch.logsumexp(log_tau, dim=-1, keepdim=True)

                tau = torch.clamp(exp(log_tau), min=1e-6) # Stabilize comutations
                tau = tau / tau.sum(dim=-1, keepdim=True) # Renormalize after stabilizing
                latent_params['transition_probs'] += [tau]

        latent_params['mask'] = mask
        latent_params['covariates'] = Cov
        latent_params['forward_means'] = torch.stack(latent_params['forward_means'], dim=1)                 # (batch_size, T-1, latent_dim)
        latent_params['bias_forward_means'] = torch.stack(latent_params['bias_forward_means'], dim=1)       # (batch_size, T-1, latent_dim)
        latent_params['covariances'] = torch.stack(latent_params['covariances'], dim=1)                     # (batch_size, T, latent_dim)
        latent_params['transition_probs'] = torch.stack(latent_params['transition_probs'], dim=1)           # (batch_size, T-1, latent_dim, switch_dim, switch_dim)

        return latent_params

    def E_step(self, state_latent_params):
        """
        E-step to compute the expectations of the latent states given the current variational parameters.
        Performs the CAVI update.
        """
        mu = []
        Sigma = []
        alpha = []
        T = state_latent_params['covariances'].shape[1]
        for t in range(T):
            if t == 0:
                mu += [state_latent_params['init_means']]
                Sigma += [torch.diag_embed(state_latent_params['covariances'][:, 0])]
                alpha += [state_latent_params['init_probs']]
            else:
                mu_prev = mu[-1]
                Sigma_prev = Sigma[-1]
                alpha_prev = alpha[-1]

                B_tilde = state_latent_params['forward_means'][:, t - 1]
                b_tilde = state_latent_params['bias_forward_means'][:, t - 1]
                psi_tilde = state_latent_params['covariances'][:, t]
                tau = state_latent_params['transition_probs'][:, t - 1]

                mu_t = B_tilde * mu_prev + b_tilde
                Sigma_t = torch.diag_embed(psi_tilde) + torch.diag_embed(B_tilde ** 2) @ Sigma_prev
                alpha_t = torch.matmul(alpha_prev.unsqueeze(-2), tau).squeeze(-2)

                mu += [mu_t]
                Sigma += [Sigma_t]
                alpha += [alpha_t]
        mu = torch.stack(mu, dim=1)        # (batch, T, latent_dim)
        Sigma = torch.stack(Sigma, dim=1)  # (batch, T, latent_dim, latent_dim)
        alpha = torch.stack(alpha, dim=1)  # (batch, T, latent_dim, switch_dim)
        expectations = {'mu': mu, 'Sigma': Sigma, 'alpha': alpha}
        return expectations

    def M_step(self, X, latent_params, expectations):
        """
        M-step updates for the model parameters given the expectations from the E-step.
        """
        # Epsilon for log stability
        eps = X.new_tensor(1e-8)
        # Compute mask for updates
        mask = latent_params['mask']                              # (batch,T)
        mask_t = mask[:, 1:].unsqueeze(-1).unsqueeze(-1)          # (batch,T-1,1,1)
        mask_star_t = mask_t.unsqueeze(-1)                        # (batch,T-1,1,1)
        # Fetch the expectation terms of the E-step, previously computed
        mu = expectations['mu']                        # (batch, T, latent_dim)
        Sigma = expectations['Sigma']                  # (batch, T, latent_dim, latent_dim)
        alpha = expectations['alpha']                  # (batch, T, latent_dim, switch_dim)
        # Compute diagonal of Sigma for ELBO computation
        sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1) # (batch, T, latent_dim)
        # Fetch the variational parameters
        nu = latent_params['init_probs']               # (batch, latent_dim, switch_dim)
        tau = latent_params['transition_probs']        # (batch, T-1, latent_dim, switch_dim, switch_dim)
        m = latent_params['init_means']                # (batch, latent_dim)
        B_tilde = latent_params['forward_means']       # (batch, T-1, latent_dim)

        # Broadcast specific parameters along switch dimension
        mu_star = mu.unsqueeze(-1).expand(-1, -1, -1, self.C)               # (batch, T, latent_dim, switch_dim)
        alpha_star = alpha.unsqueeze(-1).expand(-1, -1, -1, -1, self.C)     # (batch, T, latent_dim, switch_dim, switch_dim)
        sigma_star = sigma.unsqueeze(-1).expand(-1, -1, -1, self.C)         # (batch, T, latent_dim, switch_dim)
        m_star = m.unsqueeze(-1).expand(-1, -1, self.C)                     # (batch, latent_dim, switch_dim)
        B_tilde_star = B_tilde.unsqueeze(-1).expand(-1, -1, -1, self.C)     # (batch, T-1, latent_dim, switch_dim)
        alpha_1 = alpha[:, 0]                                               # (batch, latent_dim, switch_dim)
        alpha_star_t = alpha_star[:, :-1]                                   # (batch, T-1, latent_dim, switch_dim, switch_dim)
        alpha_t = alpha[:, 1:]                                              # (batch, T-1, latent_dim, switch_dim)

        batch_size = mu.shape[0]

        # Switching state dynamics parameters update
        self.pi = torch.clamp(nu.mean(dim=0), min=1e-6)        # (latent_dim, switch_dim)
        self.pi = self.pi / self.pi.sum(dim=-1, keepdim=True)  # (latent_dim, switch_dim)

        self.A = (mask_star_t * (tau * alpha_star_t)).sum(dim=1).sum(dim=0) / (mask_star_t * alpha_star_t).sum(dim=1).sum(dim=0).clamp_min(eps)  # (latent_dim, switch_dim, switch_dim)
        self.A = self.A.clamp_min(1e-8)
        self.A = self.A / self.A.sum(dim=-1, keepdim=True)
        # Latent state parameters update
        self.bar_b = (m_star * alpha_1).sum(dim=0) / alpha_1.sum(dim=0).clamp_min(eps) # (latent_dim, switch_dim)

        bar_b = self.bar_b.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, latent_dim, switch_dim)
        self.bar_psi = torch.clamp((alpha_1 * (sigma_star[:, 0] + (m_star - bar_b)**2)).sum(dim=0) / alpha_1.sum(dim=0).clamp_min(eps), min=1e-6)  # (latent_dim, switch_dim)

        b = self.b.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)  # (batch, 1, latent_dim, switch_dim)
        self.B = (mask_t * alpha_t * (B_tilde_star * sigma_star[:, :-1] + mu_star[:, :-1] * (mu_star[:, 1:] - b))).sum(dim=1).sum(dim=0) / (mask_t * alpha_t * (mu_star[:, :-1]**2 + sigma_star[:, :-1])).sum(dim=1).sum(dim=0).clamp_min(eps)  # (latent_dim, switch_dim)

        B = self.B.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)  # (batch, 1, latent_dim, switch_dim)
        self.b = (mask_t * alpha_t * (mu_star[:, 1:] - B * mu_star[:, :-1])).sum(dim=1).sum(dim=0) / (mask_t * alpha_t).sum(dim=1).sum(dim=0).clamp_min(eps)  # (latent_dim, switch_dim)

        B = self.B.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)  # (batch, 1, latent_dim, switch_dim)
        b = self.b.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1)  # (batch, 1, latent_dim, switch_dim)
        self.psi = torch.clamp((mask_t * alpha_t * ((mu_star[:, 1:] - B * mu_star[:, :-1] - b)**2 + sigma_star[:, 1:] + B * (B - 2 * B_tilde_star) * sigma_star[:, :-1])).sum(dim=1).sum(dim=0) / (mask_t * alpha_t).sum(dim=1).sum(dim=0).clamp_min(eps), min=1e-6)  # (latent_dim, switch_dim)
        return self

    def elbo(self, X, O, prior_params, latent_params, expectations):
        """
        Compute the Evidence Lower Bound (ELBO) of the ARPLN-ICA model.
        """

        # Retrieve batch size and device
        batch_size = X.shape[0]
        device = X.device
        # Epsilon for log stability
        eps = X.new_tensor(1e-8)

        # -----------------------------------
        # Parameter fetching and broadcasting
        # -----------------------------------
        # Mask for irregular size trajectories
        mask = latent_params['mask']                   # (batch, T)
        # Covariates
        cov = latent_params['covariates']              # (batch, cov_dim)
        # Fetch the expectation terms of the E-step, previously computed
        mu = expectations['mu']                        # (batch, T, latent_dim)
        Sigma = expectations['Sigma']                  # (batch, T, latent_dim, latent_dim)
        alpha = expectations['alpha']                  # (batch, T, latent_dim, switch_dim)
        # Compute diagonal of Sigma for ELBO computation
        sigma = torch.diagonal(Sigma, dim1=-2, dim2=-1) # (batch, T, latent_dim)
        # Fetch the variational parameters
        nu = latent_params['init_probs']               # (batch, latent_dim, switch_dim)
        tau = latent_params['transition_probs']        # (batch, T-1, latent_dim, switch_dim, switch_dim)
        m = latent_params['init_means']                # (batch, latent_dim)
        psi_tilde = latent_params['covariances']       # (batch, T, latent_dim)
        B_tilde = latent_params['forward_means']       # (batch, T-1, latent_dim)
        b_tilde = latent_params['bias_forward_means']  # (batch, T-1, latent_dim)
        # Fetch the prior parameters
        pi = prior_params['init_probs']                 # (latent_dim, switch_dim)
        A = prior_params['transition_probs']            # (latent_dim, switch_dim, switch_dim)
        bar_b = prior_params['init_means']              # (latent_dim, switch_dim)
        bar_psi = prior_params['init_covariance']       # (latent_dim, switch_dim)
        B = prior_params['forward_means']               # (latent_dim, switch_dim)
        b = prior_params['bias_forward_means']          # (latent_dim, switch_dim)
        psi = prior_params['covariances']               # (latent_dim, switch_dim)
        Gamma = prior_params['Gamma']                   # (counts_dim, latent_dim)

        # Broadcast prior parameters along batch dimension
        pi = pi.unsqueeze(0).expand(batch_size, -1, -1)               # (batch, latent_dim, switch_dim)
        A = A.unsqueeze(0).expand(batch_size, -1, -1, -1)             # (batch, latent_dim, switch_dim, switch_dim)
        bar_b = bar_b.unsqueeze(0).expand(batch_size, -1, -1)         # (batch, latent_dim, switch_dim)
        bar_psi = bar_psi.unsqueeze(0).expand(batch_size, -1, -1)     # (batch, latent_dim, switch_dim)
        B = B.unsqueeze(0).expand(batch_size, -1, -1)                 # (batch, latent_dim, switch_dim)
        b = b.unsqueeze(0).expand(batch_size, -1, -1)                 # (batch, latent_dim, switch_dim)
        psi = psi.unsqueeze(0).expand(batch_size, -1, -1)             # (batch, latent_dim, switch_dim)
        Gamma = Gamma.unsqueeze(0).expand(batch_size, -1, -1)         # (batch, counts_dim, latent_dim)

        # Broadcast specific parameters along switch dimension
        mu_star = mu.unsqueeze(-1).expand(-1, -1, -1, self.C)               # (batch, T, latent_dim, switch_dim)
        alpha_star = alpha.unsqueeze(-1).expand(-1, -1, -1, -1, self.C)     # (batch, T, latent_dim, switch_dim, switch_dim)
        sigma_star = sigma.unsqueeze(-1).expand(-1, -1, -1, self.C)         # (batch, T, latent_dim, switch_dim)
        m_star = m.unsqueeze(-1).expand(-1, -1, self.C)                     # (batch, latent_dim, switch_dim)
        B_tilde_star = B_tilde.unsqueeze(-1).expand(-1, -1, -1, self.C)     # (batch, T-1, latent_dim, switch_dim)

        # ----------------
        # ELBO computation
        # ----------------
        # ELBO initialization
        elbo = torch.tensor([0.0], device=device, dtype=X.dtype)
        for t in range(self.T):
            # Fetching time slice
            x_t = X[:, t, :]                      # (batch, counts_dim)
            mask_t = mask[:, t]                   # (batch,)
            mu_t = mu[:, t, :]                    # (batch, latent_dim)
            Sigma_t = Sigma[:, t, :]              # (batch, latent_dim, latent_dim)
            alpha_t = alpha[:, t, :]              # (batch, latent_dim, switch_dim)
            mu_star_t = mu_star[:, t, :, :]       # (batch, latent_dim, switch_dim)
            sigma_star_t = sigma_star[:, t, :, :] # (batch, latent_dim, switch_dim)
            sigma_t = torch.diagonal(Sigma_t, dim1=-2, dim2=-1)  # (batch, latent_dim)
            if O is None:
                o_t = torch.log(x_t.sum(dim=-1) + 1.).unsqueeze(-1)  # (batch, 1)
            else:
                o_t = O[:, t].unsqueeze(-1)                          # (batch, 1)
            psi_tilde_t = psi_tilde[:, t, :]                         # (batch, latent_dim)

            # Count emission term
            diag_GammaSigmaGammaT = (Gamma * Gamma * sigma_t.unsqueeze(1)).sum(dim=-1)       # (batch, counts_dim)
            Gamma_mu = torch.matmul(Gamma, mu_t.unsqueeze(-1)).squeeze(-1) + o_t + torch.matmul(self.M, cov.unsqueeze(-1)).squeeze(-1)   # (batch, counts_dim)
            exp_input = Gamma_mu + 0.5 * diag_GammaSigmaGammaT
            # Clamping for numerical stability
            exp_input = torch.clamp(exp_input, max=50.0 if exp_input.dtype == torch.float64 else 30.0)
            elbo += (mask_t*(x_t * Gamma_mu - exp(exp_input) - torch.lgamma(x_t + 1)).sum(dim=-1)).mean()
            if not torch.isfinite(elbo).all():
                raise ValueError("Non-finite ELBO encountered in emission term computation.")

            if t == 0:
                # Initial latent dynamic term
                elbo += -(nu.clamp_min(eps) * log(nu.clamp_min(eps) / pi.clamp_min(eps))).sum(dim=-1).sum(dim=-1).mean()
                # Initial state space term
                elbo += -0.5 * (alpha_t * (log(2*torch.pi*bar_psi.clamp_min(eps)) + (sigma_star_t + (m_star - bar_b)**2) / bar_psi.clamp_min(eps))).sum(dim=-1).sum(dim=-1).mean()
                if not torch.isfinite(elbo).all():
                    raise ValueError("Non-finite ELBO encountered in initial term computation.")
            else:
                # Fetching time slice for t > 0
                tau_t = tau[:, t - 1, :, :, :]                  # (batch, latent_dim, switch_dim, switch_dim)
                B_tilde_star_t = B_tilde_star[:, t - 1, :]      # (batch, latent_dim, switch_dim)
                mu_star_t_prev = mu_star[:, t - 1, :, :]        # (batch, latent_dim, switch_dim)
                sigma_star_t_prev = sigma_star[:, t - 1, :, :]  # (batch, latent_dim, switch_dim)
                alpha_star_t_prev = alpha_star[:, t - 1, :, :]  # (batch, latent_dim, switch_dim)

                # Transition latent dynamic term
                elbo += -(mask_t*(tau_t * alpha_star_t_prev * log(tau_t.clamp_min(eps) / A.clamp_min(eps))).sum(dim=-1).sum(dim=-1).sum(dim=-1)).mean()
                # Transition state space term
                # Trace term
                J = ((mu_star_t - B * mu_star_t_prev - b)**2 + sigma_star_t + B * (B - 2 * B_tilde_star_t) * sigma_star_t_prev) / psi.clamp_min(eps)
                elbo += -0.5 * (mask_t*(alpha_t * (log(2*torch.pi*psi.clamp_min(eps)) + J)).sum(dim=-1).sum(dim=-1)).mean()
                if not torch.isfinite(elbo).all():
                    raise ValueError("Non-finite ELBO encountered in forward term computation.")
            # Log determinant of psi_tilde_t, since it's diagonal we compute the sum of log diagonal elements
            elbo += 0.5 * (mask_t * log(psi_tilde_t.clamp_min(eps)).sum(dim=-1)).mean()
        # Constant terms
        T_batch = mask.sum(dim=-1)    # (batch,)
        elbo += (T_batch * self.d * log(2 * torch.pi * torch.exp(X.new_tensor(1.))) / 2).mean()
        return elbo / (T_batch.mean()*self.K*self.d*self.C**2)


    def sample(self, n_samples=1, T=None, O=None, Cov=None, seed=None):
        """
        Draw samples from the ARPLN-ICA prior model.
        Parameters
        ----------
        n_samples: int
            Number of samples to draw
        T: int, optional
            Length of the sequences to sample
        O: Tensor, optional
            Offsets to condition the emission distribution on (n_samples, T)
        Cov: Tensor, optional
            Covariates to condition the emission distribution on (n_samples, cov_dim)
        seed: int, optional
            Random seed for reproducibility

        Returns
        -------
        X: Tensor
            Sampled counts (n_samples, T, K)
        S: Tensor
            Sampled states (n_samples, T, d)
        U: Tensor
            Sampled switching states (n_samples, T, d)
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        if T is None:
            T = self.T
        if O is None:
            # Select random starting offsets from training data
            rand_indices = torch.randint(0, self.n, (n_samples,))
            O = self.offsets[rand_indices, :].to(self.device, dtype=self.counts.dtype).unsqueeze(-1)  # (n_samples, T, 1)
        if Cov is None:
            # Select random covariates from training data
            rand_indices = torch.randint(0, self.n, (n_samples,))
            Cov = self.covariates[rand_indices, :].to(self.device, dtype=self.counts.dtype)  # (n_samples, cov_dim)
        with torch.no_grad():
            # Initialize tensors to hold samples
            u = torch.zeros(n_samples, T, self.d).to(self.device, dtype=self.counts.dtype)
            s = torch.zeros(n_samples, T, self.d).to(self.device, dtype=self.counts.dtype)
            x = torch.zeros(n_samples, T, self.K).to(self.device, dtype=self.counts.dtype)

            for t in range(T):
                if t == 0:
                    for i in range(self.d):
                        # Draw initial switching states for dimension i u^{(i)}_1
                        u[:, 0, i] = torch.distributions.Categorical(probs=self.pi[i, :]).sample((n_samples,)).to(self.device, dtype=self.counts.dtype)
                        # Draw initial latent states for dimension i s^{(i)}_1 conditionally on u^{(i)}_1
                        s[:, 0, i] = torch.distributions.Normal(
                            loc=self.bar_b[i, u[:, 0, i].long()],
                            scale=torch.sqrt(self.bar_psi[i, u[:, 0, i].long()])
                        ).sample().to(self.device, dtype=self.counts.dtype)
                else:
                    for i in range(self.d):
                        u_prev = u[:, t - 1, i].long()
                        # Draw switching states for dimension i u^{(i)}_t conditionally on u^{(i)}_{t-1}
                        u[:, t, i] = torch.distributions.Categorical(probs=self.A[i, u_prev, :]).sample().to(self.device, dtype=self.counts.dtype)
                        # Draw latent states for dimension i s^{(i)}_t conditionally on s^{(i)}_{t-1} and u^{(i)}_t
                        s_prev = s[:, t - 1, i]
                        forward_mean = self.B[i, u[:, t, i].long()] * s_prev + self.b[i, u[:, t, i].long()]
                        s[:, t, i] = torch.distributions.Normal(
                            loc=forward_mean,
                            scale=torch.sqrt(self.psi[i, u[:, t, i].long()])
                        ).sample().to(self.device, dtype=self.counts.dtype)
                # Draw counts x_t conditionally on s_t
                x[:, t] = self.emission_sample(s[:, t], O_t=O[:, t], Cov=Cov)
        return x, s, u

    def vamp_sample(self, n_samples=1, X=None, O=None, Cov=None, times=None, seed=None):
        """
        Draw samples from the variational distribution using VAMP-style sampling.
        Parameters
        ----------
        n_samples: int
            Number of samples to draw
        X: Tensor, optional
            Input data to condition the variational distribution on (n_samples, T, K)
        O: Tensor, optional
            Offsets to condition the variational distribution on (n_samples, T)
        Cov: Tensor, optional
            Covariates to condition the variational distribution on (n_samples, cov_dim)
        times: Tensor, optional
            Time points for irregular data (n_samples, T)
        seed: int, optional
            Random seed for reproducibility

        Returns
        -------
        X: Tensor
            Sampled counts (n_samples, T, K)
        S: Tensor
            Sampled states (n_samples, T, d)
        U: Tensor
            Sampled switching states (n_samples, T, d)
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        with torch.no_grad():
            # Randomly select n_samples from the training data
            if X is None:
                indices = torch.randint(0, self.n, (n_samples,))
                X_s = self.counts[indices, :, :]
                O = self.offsets[indices, :].to(self.device, dtype=self.counts.dtype)       # (n_samples, T)
                Cov = self.covariates[indices, :].to(self.device, dtype=self.counts.dtype)  # (n_samples, cov_dim)
            else:
                X_s = X.to(self.device, dtype=self.counts.dtype)
            if O is None:
                O = torch.log(X_s.sum(dim=-1) + 1.).to(self.device, dtype=self.counts.dtype)  # (n_samples, T)
            if Cov is None:
                Cov = self.covariates[indices, :].to(self.device, dtype=self.counts.dtype)    # (n_samples, cov_dim)
            if times is None:
                times = torch.full((n_samples, X_s.shape[1]), 1).to(self.device, dtype=self.counts.dtype)
            n_samples = X_s.shape[0]
            # Compute the variational parameters for the selected data
            latent_params = self.forward(X_s, O, Cov=Cov, times=times)
            nu = latent_params['init_probs']                # (batch, latent_dim, switch_dim)
            tau = latent_params['transition_probs']         # (batch, T-1, latent_dim, switch_dim, switch_dim)
            m = latent_params['init_means']                 # (batch, latent_dim)
            psi_tilde = latent_params['covariances']        # (batch, T, latent_dim)
            B_tilde = latent_params['forward_means']        # (batch, T-1, latent_dim)
            b_tilde = latent_params['bias_forward_means']   # (batch, T-1, latent_dim)

            # Sample from the variational distribution
            T = X_s.shape[1]
            s = torch.zeros(n_samples, T, self.d).to(self.device, dtype=self.counts.dtype)
            u = torch.zeros(n_samples, T, self.d).to(self.device, dtype=self.counts.dtype)
            x = torch.zeros(n_samples, T, self.K).to(self.device, dtype=self.counts.dtype)
            for t in range(T):

                if t == 0:
                    s[:, t] = torch.distributions.MultivariateNormal(
                        loc=m,
                        covariance_matrix=torch.diag_embed(psi_tilde[:, 0, :])
                    ).sample().to(self.device, dtype=self.counts.dtype)

                    for i in range(self.d):
                        u_i = torch.distributions.Categorical(probs=nu[:, i, :]).sample().to(self.device)
                        u[:, t, i] = u_i
                else:
                    s[:, t] = torch.distributions.MultivariateNormal(
                        loc=B_tilde[:, t - 1, :] * s[:, t - 1, :] + b_tilde[:, t - 1, :],
                        covariance_matrix=torch.diag_embed(psi_tilde[:, t, :])
                    ).sample().to(self.device, dtype=self.counts.dtype)

                    for i in range(self.d):
                        for sample in range(n_samples):
                            u_prev = u[sample, t - 1, i].long()
                            u_i = torch.distributions.Categorical(probs=tau[sample, t - 1, i, u_prev, :]).sample().to(self.device, dtype=self.counts.dtype)
                            u[sample, t, i] = u_i

                # Draw counts x_t conditionally on s_t
                x[:, t] = self.emission_sample(s[:, t], O_t=O[:, t].unsqueeze(-1), Cov=Cov)
        return x, s, u

    def identify(self, Gamma_ref):
        """
        Align the learned mixing matrix Gamma to a reference mixing matrix Gamma_ref
        using permutation and sign flips to minimize the distance between them.
        Parameters
        ----------
        Gamma_ref: torch.Tensor
            Reference mixing matrix of shape (K, d)

        Returns
        -------
        Gamma_aligned: torch.Tensor
            Aligned mixing matrix of shape (K, d)
        best_perm: torch.Tensor
            Permutation indices of shape (d,)
        sign: torch.Tensor
            Sign flips of shape (d,)
        best_score: float
            Alignment score (1-cosine similarity, lower is better)
        """
        Gamma = self.Gamma.normalize()  # (K, d)
        # cosine similarity matrix since columns are unit norm
        S = Gamma_ref.T @ Gamma  # (d, d)
        d = S.shape[0]

        best_score = None
        best_perm = None
        for p in itertools.permutations(range(d)):
            p = torch.tensor(p, device=S.device)
            score = S[torch.arange(d, device=S.device), p].abs().sum()
            if (best_score is None) or (score > best_score):
                best_score = score
                best_perm = p

        # sign per matched column (make dot product positive)
        diag = S[torch.arange(d, device=S.device), best_perm]
        sign = torch.sign(diag)
        sign[sign == 0] = 1.0

        Gamma_perm = Gamma[:, best_perm]  # reorder
        Gamma_aligned = Gamma_perm * sign.unsqueeze(0)  # flip signs

        best_score = 1 - (best_score / Gamma_ref.shape[1]).item()
        return Gamma_aligned.detach(), best_perm, sign, best_score


    def predict_vamp(self, X, T=None, O=None, Cov=None, times=None, viterbi=True):
        """
        Predict future observations using VAMP-style prediction for switching labels.
        Only runs if predictive mode was used in training.
        Parameters
        ----------
        X: Tensor
            Input data to condition the variational distribution on (batch, T_past, K)
        T: int, optional
            Total length of the sequences to predict (including observed part)
        O: Tensor, optional
            Offsets to condition the variational distribution on (batch, T)
        Cov: Tensor, optional
            Covariates to condition the variational distribution on (batch, cov_dim)
        times: Tensor, optional
            Time points for irregular data (batch, T_past)
        viterbi: bool, optional
            Whether to use Viterbi algorithm for MAP switching state prediction

        Returns
        -------
        Dictionary with keys:
            prediction: torch.Tensor
                Predicted mean counts (batch, T, K)
            variance: torch.Tensor
                Predicted counts variance (batch, T, K)
            hat_alpha: torch.Tensor
                Marginal switching state probabilities per subject (batch, T, d, C)
            bar_alpha: torch.Tensor
                Marginal switching state probabilities from training data (T, d, C)
            bar_tau: torch.Tensor
                Marginal switching kernels from training data (T-1, d, C, C)
            hat_mu: torch.Tensor
                Predicted latent states (batch, T, d)
            hat_psi: torch.Tensor
                Predicted latent state covariances (batch, T, d)
        """
        t_0 = X.shape[1]
        batch_size = X.shape[0]

        if T is None or T > self.T:
            T = self.T
        if T < t_0:
            print("WARNING: T < observed length, no need for prediction.")
            return X

        X = X.to(self.device, dtype=self.counts.dtype)
        if O is None:
            O_obs = torch.log(X.sum(dim=-1) + 1.0).to(self.device, dtype=self.counts.dtype)  # (batch, T_past)
            O_future = O_obs[:, -1].unsqueeze(-1).expand(-1, T - t_0)  # hold last offset
            O = torch.cat([O_obs, O_future], dim=1)             # (batch, T)
        if Cov is None:
            Cov = self.covariates[:1].expand(batch_size, -1).to(self.device, dtype=self.counts.dtype)  # (batch, cov_dim)
        if times is None:
            times = self.times[:1].expand(batch_size, -1).to(self.device, dtype=self.counts.dtype)  # (batch, T_past)

        with torch.no_grad():
            # ----------------------------
            # Computation of VAMP switching
            # ---------------------------
            # Compute variational parameters on training data
            latent_params_bank = self.forward(self.counts, self.offsets, Cov=self.covariates, times=self.times)
            expectations_bank = self.E_step(latent_params_bank)

            alpha_bank = expectations_bank['alpha']               # (n_train, T, d, C)
            tau_bank = latent_params_bank['transition_probs']     # (n_train, T-1, d, C, C)

            # Expand alpha for computation of bar_alpha and bar_tau
            alpha_star_prev_bank = alpha_bank.unsqueeze(-1).expand(-1, -1, -1, -1, self.C)[:, :-1]   # (n_train, T-1, d, C, C)

            # Compute marginal bar_alpha, the inhomogeneous marginal switching state probability
            bar_alpha = alpha_bank.mean(dim=0)                                                        # (T, d, C)
            # Compute bar_tau, the inhomogeneous switching kernels
            bar_tau = (tau_bank * alpha_star_prev_bank).sum(dim=0) / alpha_star_prev_bank.sum(dim=0)  # (T-1, d, C, C)

            # ----------------------------
            # Latent state prediction with VAMP
            # ---------------------------
            # Compute variational parameters on observed counts
            latent_params = self.forward(X, O[:, :t_0], Cov=Cov, times=times)
            expectations = self.E_step(latent_params)
            alpha = expectations['alpha']                           # (batch, T_past, d, C)
            tau = latent_params['transition_probs']                 # (batch, T_past-1, d, C, C)
            mu = expectations['mu']                                 # (batch, T_past, d)
            tilde_psi = latent_params['covariances']                # (batch, T_past, d)
            B = self.B.unsqueeze(0).expand(batch_size, -1, -1)     # (batch, d, C)
            b = self.b.unsqueeze(0).expand(batch_size, -1, -1)     # (batch, d, C)
            psi = self.psi.unsqueeze(0).expand(batch_size, -1, -1) # (batch, d, C)

            # Compute hat_alpha recursion
            hat_alpha = torch.zeros(batch_size, T, self.d, self.C, device=self.device, dtype=self.counts.dtype)
            hat_alpha[:, :t_0] = alpha
            for t in range(t_0, T):
                bar_tau_t = bar_tau[t-1].unsqueeze(0).expand(batch_size, -1, -1, -1)                  # (batch_size, d, C, C)
                hat_alpha_t_prev = hat_alpha[:, t-1]                                                  # (batch_size, d, C)
                hat_alpha[:, t] = torch.matmul(hat_alpha_t_prev.unsqueeze(-2), bar_tau_t).squeeze(-2) # (batch_size, d, C)

            # Draw Viterbi MAP trajectory if selected
            if viterbi:
                hat_tau = bar_tau.unsqueeze(0).expand(hat_alpha.shape[0], -1, -1, -1, -1)
                hat_tau[:, :t_0-1] = tau 
                hat_alpha = viterbi_algorithm(alpha=hat_alpha, tau=hat_tau)
            
        
            # Compute hat_mu, hat_psi recursion
            hat_mu = torch.zeros(batch_size, T, self.d, device=self.device, dtype=self.counts.dtype)
            hat_psi = torch.zeros(batch_size, T, self.d, device=self.device, dtype=self.counts.dtype)

            # Initialize with observed part
            hat_mu[:, :t_0] = mu
            hat_psi[:, :t_0] = tilde_psi
            for t in range(t_0, T):
                hat_mu_star_prev = hat_mu[:, t-1].unsqueeze(-1).expand(-1, -1, self.C)         # (batch, d, C)
                hat_mu[:, t] = (hat_alpha[:, t] * (B * hat_mu_star_prev + b)).sum(dim=-1)      # (batch, d)

                hat_psi_star_prev = hat_psi[:, t-1].unsqueeze(-1).expand(-1, -1, self.C) # (batch, d, C)
                hat_psi[:, t] = (hat_alpha[:, t] * (psi + B**2 * hat_psi_star_prev + (B * hat_mu_star_prev + b)**2)).sum(dim=-1) - hat_mu[:, t]**2

            # ----------------------------
            # Counts prediction
            # ---------------------------
            x_mean = torch.zeros(batch_size, T, self.K, device=self.device, dtype=self.counts.dtype)
            x_var = torch.zeros(batch_size, T, self.K, device=self.device, dtype=self.counts.dtype)

            Gamma = self.Gamma.normalize().unsqueeze(0)  # (1, K, d)

            for t in range(T):
                diag_GammaSigmaGammaT = (Gamma**2 * hat_psi[:, t].unsqueeze(1)).sum(dim=-1)  # (batch, K)
                Gamma_mu = torch.matmul(Gamma, hat_mu[:, t].unsqueeze(-1)).squeeze(-1) + O[:, t].unsqueeze(-1) + torch.matmul(self.M.unsqueeze(0), Cov.unsqueeze(-1)).squeeze(-1)  # (batch, K)
                exp_input = Gamma_mu + 0.5 * diag_GammaSigmaGammaT
                # Clamping for numerical stability
                exp_input = torch.clamp(exp_input, max=50.0 if exp_input.dtype == torch.float64 else 30.0)
                x_mean[:, t] = torch.exp(exp_input)
                x_var[:, t] = x_mean[:, t] + x_mean[:, t]**2 * (torch.exp(diag_GammaSigmaGammaT) - 1.0)

            return {
                'prediction': x_mean,
                'variance': x_var,
                'hat_alpha': hat_alpha,
                'bar_alpha': bar_alpha,
                'bar_tau': bar_tau,
                'hat_mu': hat_mu,
                'hat_psi': hat_psi
            }

    def predict_prior(self, X, T=None, O=None, Cov=None, times=None, seed=None):
        """
        Predict future observations given past data under ARPLN-ICA prior model.

        Parameters
        ----------
        X: torch.Tensor
            Input count data of shape (batch, T_past, counts_dim)
        T: int, optional
            Total length of the sequences to predict (including observed part)
        O: torch.Tensor, optional
            Offsets to condition the emission distribution on (batch, T)
        Cov: torch.Tensor, optional
            Covariates to condition the emission distribution on (batch, cov_dim)
        times: torch.Tensor, optional
            Time points for irregular data (batch, T_past)
        seed: int, optional
            Random seed for reproducibility

        Returns
        -------
        X: torch.Tensor
            Predicted count data of shape (batch, T, counts_dim)
        S: torch.Tensor
            Predicted latent states of shape (batch, T, latent_dim)
        U: torch.Tensor
            Predicted switching states of shape (batch, T, latent_dim)
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        if not self.predictive:
            raise ValueError("Model was not trained for prediction. Set predictive=True when initializing the model.")

        X = X.to(self.device, dtype=self.counts.dtype)
        n_samples = X.shape[0]
        T_past = X.shape[1]
        if T is None:
            T = self.T
        T_future = T - T_past
        if T_future < 0:
            return X

        if O is None:
            O_past = torch.log(X.sum(dim=-1) + 1.).to(self.device, dtype=self.counts.dtype)  # (n_samples, T_past)
            # Assume future offsets are the same as the last observed offset
            O_future = O_past[:, -1].unsqueeze(-1).expand(-1, T_future)  # (n_samples, T - T_past)
            O = torch.cat([O_past, O_future], dim=1).unsqueeze(-1)  # (n_samples, T, 1)
        else:
            O = O.to(self.device, dtype=self.counts.dtype)
            assert O.shape[0] == n_samples and O.shape[1] == T, "Offsets O must have shape (n_samples, T)."
            O = O.unsqueeze(-1)  # (n_samples, T, 1)

        X_pred = torch.zeros(n_samples, T, self.K).to(self.device, dtype=self.counts.dtype)
        X_pred[:, :T_past, :] = X
        s_path = torch.zeros(n_samples, T, self.d).to(self.device, dtype=self.counts.dtype)
        u_path = torch.zeros(n_samples, T, self.d).to(self.device, dtype=self.counts.dtype)

        with torch.no_grad():
            # Initialize from variational posterior given the observed past
            _, s_q, u_q = self.vamp_sample(X=X, O=O[:, :T_past, 0], Cov=Cov, times=times, seed=seed)
            # Store state paths
            s_path[:, :T_past, :] = s_q
            u_path[:, :T_past, :] = u_q
            # Initialize previous states
            s_prev = s_q[:, -1, :]
            u_prev = u_q[:, -1, :].long()

            for t in range(T_past, T):
                # Sample switching states from the prior transition using previous switching state
                u_t = torch.zeros(n_samples, self.d).to(self.device, dtype=self.counts.dtype)
                for i in range(self.d):
                    u_t[:, i] = torch.distributions.Categorical(
                        probs=self.A[i, u_prev[:, i], :],
                    ).sample().to(self.device, dtype=self.counts.dtype)
                u_t = u_t.long()

                # Sample latent states from the prior transition using previous state and current switching state
                s_t = torch.zeros_like(s_prev).to(self.device, dtype=self.counts.dtype)
                for i in range(self.d):
                    forward_mean = self.B[i, u_t[:, i]] * s_prev[:, i] + self.b[i, u_t[:, i]]
                    var = torch.sqrt(self.psi[i, u_t[:, i]])
                    s_t[:, i] = torch.distributions.Normal(
                        loc=forward_mean,
                        scale=var,
                    ).sample().to(self.device, dtype=self.counts.dtype)

                # Emit observation
                X_pred[:, t] = self.emission_sample(s_t, O_t=O[:, t], Cov=Cov)

                # Store paths and update previous states
                s_path[:, t, :] = s_t
                u_path[:, t, :] = u_t
                s_prev = s_t
                u_prev = u_t

        return X_pred, s_path, u_path

    def emission_sample(self, s_t, O_t, Cov):
        """
        Sample counts from the emission distribution given latent states s_t, offsets O_t, and covariates Cov.
        Parameters
        ----------
        s_t: Tensor
            Latent states at time t (n_samples, latent_dim)
        O_t: Tensor
            Offsets at time t (n_samples, 1)
        Cov: Tensor
            Covariates (n_samples, cov_dim)

        Returns
        -------
        x_t: Tensor
            Sampled counts (n_samples, counts_dim)
        """
        n_samples = s_t.shape[0]
        Gamma = self.Gamma.normalize().unsqueeze(0).expand(n_samples, -1, -1)
        M = self.M.unsqueeze(0).expand(n_samples, -1, -1)
        z_t = (Gamma @ s_t.unsqueeze(-1) + O_t.unsqueeze(-1) + M @ Cov.unsqueeze(-1)).squeeze(-1)  # (n_samples, counts_dim)
        # Then, we compute lambda_t and sample the counts
        lambda_t = torch.exp(z_t.clamp(max=40.))  # (n_samples, counts_dim)
        return torch.distributions.Poisson(rate=lambda_t).sample().to(self.device, dtype=self.counts.dtype)

    def latent_states_params(self, X, O=None, Cov=None, times=None):
        """
        Compute the parameters of the latent states variational distribution given observed counts X.
        Parameters
        ----------
        X: Tensor
            Input data to condition the variational distribution on (batch, T, K)
        O: Tensor, optional
            Offsets to condition the emission distribution on (batch, T)
        Cov: Tensor, optional
            Covariates to condition the emission distribution on (batch, cov_dim)
        times: Tensor, optional
            Time points for irregular data (batch, T)

        Returns
        -------
        mu: Tensor
            Mean of the latent states variational distribution (batch, T, d)
        Sigma: Tensor
            Covariance of the latent states variational distribution (batch, T, d, d)
        """
        self.eval()
        with torch.no_grad():
            X = X.to(self.device, dtype=self.counts.dtype)
            batch_size, T, K = X.shape

            if times is None:
                times = torch.arange(T, device=self.device, dtype=self.counts.dtype).unsqueeze(0).expand(batch_size, -1)
            else:
                times = times.to(self.device, dtype=self.counts.dtype)
                assert times.shape == (batch_size, T), "times must have shape (batch, T)."

            if O is None:
                O = torch.log(X.sum(dim=-1) + 1.).to(self.device, dtype=self.counts.dtype)  # (batch,T)
            else:
                O = O.to(self.device, dtype=self.counts.dtype)
                if O.dim() == 3 and O.shape[-1] == 1:
                    O = O[..., 0]
                assert O.shape == (batch_size, T), "O must have shape (batch, T) (or (batch,T,1))."

            latent_params = self.forward(X, O, Cov=Cov, times=times)
            expectations = self.E_step(latent_params)
            mu = expectations['mu']                                               # (batch, T, d)
            Sigma_diag = torch.diagonal(expectations['Sigma'], dim1=-2, dim2=-1)  # (batch, T, d)

            return mu, Sigma_diag

    def log_intensity(self, X, O=None, Cov=None, times=None):
        """
        Compute the log-intensity (Poisson log parameter) given observed counts X.
        Parameters
        ----------
        X: Tensor
            Input data to condition the variational distribution on (n_samples, T, K)
        O: Tensor, optional
            Offsets to condition the emission distribution on (n_samples, T)
        Cov: Tensor, optional
            Covariates to condition the emission distribution on (n_samples, cov_dim)
        times: Tensor, optional
            Time points for irregular data (n_samples, T)

        Returns
        -------
        z: Tensor
            Log-intensity (n_samples, T, K)
        """
        mu, _ = self.latent_states_params(X, O=O, Cov=Cov, times=times)
        n_samples = X.shape[0]
        Gamma = self.Gamma.normalize().unsqueeze(0).expand(n_samples, -1, -1)
        M = self.M.unsqueeze(0).expand(n_samples, -1, -1)
        if O is None:
            O = torch.log(X.sum(dim=-1, keepdims=True) + 1.).to(self.device, dtype=self.counts.dtype)  # (n_samples, T)
        else:
            O = O.to(self.device, dtype=self.counts.dtype)
            if O.dim() == 3 and O.shape[-1] == 1:
                O = O[..., 0]
            assert O.shape[0] == n_samples and O.shape[1] == X.shape[1], "Offsets O must have shape (n_samples, T)."
        if Cov is None:
            Cov = self.covariates[:n_samples, :].to(self.device, dtype=self.counts.dtype)  # (n_samples, cov_dim)
        else:
            Cov = Cov.to(self.device, dtype=self.counts.dtype)
            assert Cov.shape[0] == n_samples and Cov.shape[1] == self.covariates.shape[1], "Covariates must have shape (n_samples, cov_dim)."
        z = torch.zeros_like(X)
        for t in range(z.shape[1]):
            z[:, t] = (Gamma @ mu[:, t].unsqueeze(-1) + O[:, t].unsqueeze(-1) + M @ Cov.unsqueeze(-1)).squeeze(-1)  # (n_samples, T, counts_dim)
        return z

    def reconstruction(self, X, O=None, Cov=None, times=None):
        """
        Compute the log-reconstruction of observed counts X through ARPLN-ICA (smoothing).
        Parameters
        ----------
        X: Tensor
            Input data to condition the variational distribution on (n_samples, T, K)
        O: Tensor, optional
            Offsets to condition the emission distribution on (n_samples, T)
        Cov: Tensor, optional
            Covariates to condition the emission distribution on (n_samples, cov_dim)
        times: Tensor, optional
            Time points for irregular data (n_samples, T)

        Returns
        -------
        z: Tensor
            Log-reconstruction (n_samples, T, K)
        """
        mu, sigma = self.latent_states_params(X, O=O, Cov=Cov, times=times)
        n_samples = X.shape[0]
        Gamma = self.Gamma.normalize().unsqueeze(0).expand(n_samples, -1, -1)
        M = self.M.unsqueeze(0).expand(n_samples, -1, -1)
        if O is None:
            O = torch.log(X.sum(dim=-1, keepdims=True) + 1.).to(self.device, dtype=self.counts.dtype)  # (n_samples, T)
        else:
            O = O.to(self.device, dtype=self.counts.dtype)
            if O.dim() == 3 and O.shape[-1] == 1:
                O = O[..., 0]
            assert O.shape[0] == n_samples and O.shape[1] == X.shape[1], "Offsets O must have shape (n_samples, T)."
        if Cov is None:
            Cov = self.covariates[:n_samples, :].to(self.device, dtype=self.counts.dtype)  # (n_samples, cov_dim)
        else:
            Cov = Cov.to(self.device, dtype=self.counts.dtype)
            assert Cov.shape[0] == n_samples and Cov.shape[1] == self.covariates.shape[
                1], "Covariates must have shape (n_samples, cov_dim)."
        z = torch.zeros_like(X)
        for t in range(z.shape[1]):
            z[:, t] = (Gamma @ mu[:, t].unsqueeze(-1) + O[:, t].unsqueeze(-1) + M @ Cov.unsqueeze(-1)).squeeze(-1)  # (n_samples, T, counts_dim)
            z[:, t] += torch.diagonal(Gamma @ torch.diag_embed(sigma[:, t]) @ Gamma.mT, dim1=-1, dim2=-2) / 2
        return z

    def to_cpu(self):
        """
        Move the model parameters to CPU.
        """
        self.device = torch.device('cpu')
        self.to(device='cpu')
        self.pi = self.pi.cpu()
        self.A = self.A.cpu()
        self.M = self.M.cpu()
        self.bar_b = self.bar_b.cpu()
        self.bar_psi = self.bar_psi.cpu()
        self.B = self.B.cpu()
        self.b = self.b.cpu()
        self.psi = self.psi.cpu()
        self.D = self.D.cpu()
        self.P = self.P.cpu()
        if hasattr(self, 'counts'):
            self.counts = self.counts.cpu()
        if hasattr(self, 'offsets'):
            self.offsets = self.offsets.cpu()
        if hasattr(self, 'covariates'):
            self.covariates = self.covariates.cpu()
        if hasattr(self, 'times'):
            self.times = self.times.cpu()
        return self