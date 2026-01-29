# Independent Component Discovery in Temporal count data

This repository contains codes for reproducing the experiments of *Independent Component Discovery in Temporal count data*. 
In particular, this repository provides an implementation of ARPLN-ICA models.

Typical applications involve:
- **Independent Component Analysis (ICA)** for temporal count data: co-variation patterns discovery and characterization of dynamics.
- **Dimensionality reduction** of temporal count data through ICA.
- **Perturbation modeling** in temporal count data.
- **Data generation** of temporal count data with forecasting.

# 🛠 Installation
The code is implemented in Python 3.11.10. If you only plan on using ARPLN-ICA models, you can install the minimum required dependencies using pip:
```py
pip install torch==2.5.1;
pip install numpy==1.26.4;
pip install tqdm==4.67.1;
```

If you plan on reproducing the experiments, you need to install additional dependencies packaged in `requirements.txt`:
```py
pip install -r requirements.txt
```
You also need to install [MIMIC](https://github.com/ucl-cssb/MIMIC) to run Lotka-Volterra mechanistic baselines (not available through pip).

# ⚡️ Quickstart

This package provides an implementation of ARPLN-ICA models through the `PLNICA` class. 
Here are a few example of usages.

```py
from plntime.models import PLNICA

# Temporal count data (n_samples x n_timepoints x n_features) 
# predefined as a torch.Tensor X

# Initialize model
model = PLNICA(
    counts=X,
    latent_size=5,    # Number of independent components
    n_dynamics=2,     # Number of latent regimes
    predictive=False, #  Use model for forecasting if True
)

# Fit the model
loss = model.fit(n_epochs=800)

# Fetch the linear mixing matrix for ICA analysis
mixing_matrix = model.Gamma.normalize()

# Transform data to independent components space (mean and variance of latent states)
s_mean, s_var = model.latent_states_params(X)

# Regime inference per sample, components and timepoints
# Marginal probability tensor of shape (n_samples x n_timepoints x n_components x n_dynamics)
hat_alpha = model.predict_vamp(X)['hat_alpha']

# Smooth observed counts
x_log_smooth = model.reconstruction(X)

# Forecast future counts (if predictive=True)
t_0 = 5     # Starting timepoint for forecasting
predictions = model.predict_vamp(X[:, :t_0])
x_pred_mean = predictions['prediction']
x_pred_var  = predictions['prediction_var']
```

# 🔍 Available experiments

## Simulation studies

The notebook `simulated_study.ipynb` contains the experiments of the original paper on 3 simulated settings.

The goal is to evaluate the capacity of ARPLN-ICA to recover the ICA mixing function in a finite-sample regime.
Comparison of two inference strategies is performed (AR/MF), as well as comparison with other ICA methods (UwedgeICA/Picard).

Additional experiments show the impact of the number of samples on the recovery performance, as well as wall-clock timings.


## Microbiome analysis illustration

The notebook `microbiome_analysis.ipynb` contains the experiments of the original paper on a gut microbiome dataset from [Bucci et al. (2016)](https://doi.org/10.1186/s13059-016-0980-6).

The goal is to identify independent components of co-varying microbial taxa in the gut microbiome of mice undergoing an infection with *C. difficile*.
Results highlight biologically relevant components, as well as perturbation models aligned with clinical perturbations.

The notebook also includes an auxiliary forecasting experiment, showing the capacity of ARPLN-ICA to forecast temporal count data comparatively to standard approaches.

# 📜 Citations
Please cite our work using the following reference:
```
Chaussard, A., Bonnet, A., Le Corff, S.. Independent Component Discovery in Temporal Count data. arXiv preprint (2026).
```
