import pandas as pd

def load_data(path='plntime/data/bucci_gnotobiotic_mice/'):
    """
    Load the data from the MDSINE dataset.
    The data includes time series of microbiome compositions for 5 mice controlled over 28 days.

    Bucci, V., Tzen, B., Li, N., Simmons, M., Tanoue, T., Bogart, E., ... & Gerber, G. K. (2016).
    MDSINE: Microbial Dynamical Systems INference Engine for microbiome time-series analyses. Genome biology, 17, 1-17.
    Parameters
    ----------
    path

    Returns
    -------

    """

    return pd.read_csv(path + 'longitudinal_data.csv')
