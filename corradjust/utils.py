import pandas as pd
import numpy as np

from numba import njit, prange
from scipy.stats import beta


# Global constants for numba
SMALLEST_FLOAT64 = np.finfo(np.float64).smallest_normal

######################################################
# Classes for training/test-friendly data processing #
######################################################


class MeanCenterizer:
    """
    Center data by subtracting mean value from each feature.
    This class has two important features.

    - It is train/test friendly.
    - It can separately centralize different groups of samples,
      thus residualizing group effect from input data.
    
    Attributes
    ----------
    all_features : list
        List of all features of the training set.
    means : dict
        A dict with keys being group names, and values being
        numpy arrays with feature-wise training set means.
        Populated by calling `fit` method.
        If `df_samp_ann` passed to `fit` is ``None``,
        `means` will have a single key: ``"all_samples"``.
    """

    def fit(self, df_data, df_samp_ann=None):
        """
        Compute and save mean values for centering
        in the `means` attribute.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame providing group annotation of samples. If ``None``,
            means across all samples will be computed.
            
            - Index: sample names (should match index of `df_data`).
            - Column group: discrete set of group names (e.g., ``"normal"`` and ``"tumor"``).
        """

        self.all_features = df_data.columns

        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)
        
        self.means = {}
        for samp_group in df_samp_ann["group"].unique():
            self.means[samp_group] = df_data.loc[df_samp_ann["group"] == samp_group].to_numpy().mean(axis=0)
    
    def transform(self, df_data, df_samp_ann=None):
        """
        Center the data using mean values computed
        during `fit` call.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame providing group annotation of samples.
            See `fit` for the format.
        
        Returns
        -------
        pandas.DataFrame
            Mean-centered version of `df_data`.
        """
        
        assert np.all(df_data.columns == self.all_features)
        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)

        df_data_centered = df_data.copy()
        for samp_group in df_samp_ann["group"].unique():
            group_mask = df_samp_ann["group"] == samp_group
            df_data_centered.loc[group_mask] -= self.means[samp_group]

        return df_data_centered 


class Winsorizer:
    """
    Winsorize each feature.
    This class is train/test friendly.

    Parameters
    ----------
    alpha : float, optional, default=0.05
        For each feature, `alpha` fraction of the lowest values
        and `alpha` fraction of the highest values
        will be changed to the corresponding quantile values.
    
    Attributes
    ----------
    alpha : float
        Populated by the constructor.
    all_features : list
        List of all features of the training set.
    lower_thresholds, upper_thresholds : numpy.array
        Arrays with feature-wise training set
        ``alpha`` and ``1 - alpha`` quantiles.
        Populated by calling `fit` method.
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def fit(self, df_data):
        """
        Compute and save winsorization thresholds.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
        """

        self.all_features = df_data.columns
        # This setting of interpolation will guarantee equal number of winsorized elements
        self.lower_thresholds = df_data.quantile(self.alpha, interpolation="lower", axis=0).to_numpy()
        self.upper_thresholds = df_data.quantile(1 - self.alpha, interpolation="higher", axis=0).to_numpy()
    
    def transform(self, df_data):
        """
        Winsorize the data using threshold values computed
        during `fit` call.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
        
        Returns
        -------
        pandas.DataFrame
            Winsorized version of `df_data`.
        """

        assert np.all(df_data.columns == self.all_features)

        # Duplicate threshold across rows to match df_data shape
        lower = np.tile(self.lower_thresholds[:, None], df_data.shape[0]).T
        upper = np.tile(self.upper_thresholds[:, None], df_data.shape[0]).T

        df_data_wins = df_data.copy()
        df_data_wins[:] = np.where(df_data_wins <= lower, lower, df_data_wins)
        df_data_wins[:] = np.where(df_data_wins >= upper, upper, df_data_wins)

        return df_data_wins 


class MedianOfRatios:
    """
    Median of ratios normalization for RNA-seq data.
    This class is train/test friendly.
    
    Attributes
    ----------
    all_features : list
        List of all features of the training set.
    geo_means : numpy.array
        Array with gene-wise training set geometric means.
        Populated by calling `fit` method.
    """

    def fit(self, df_data):
        """
        Compute and save gene-wise geometrical means.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Gene expression table. Index = samples, columns = gene ids.
        """

        self.all_features = df_data.columns
        # Remove genes with at least one zero
        df_data = df_data.loc[:, df_data.min(axis=0) > 0]
        # Geometric means of genes without zeros
        self.geo_means = 2 ** np.log2(df_data).mean(axis=0)
    
    def transform(self, df_data):
        """
        Normalize the data using geometric mean values computed
        during `fit` call.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Gene expression table. Index = samples, columns = gene ids.
        
        Returns
        -------
        pandas.DataFrame
            Normalized version of `df_data`.
        """

        assert np.all(df_data.columns == self.all_features)

        # Divide each sample by training set mean
        ratios = df_data[self.geo_means.index] / self.geo_means
        # Compute median of ratios
        size_factors = ratios.median(axis=1)
        # Divide each sample by its own size factor
        df_data_norm = df_data.divide(size_factors, axis=0)

        return df_data_norm


def make_dummy_samp_ann(df_data):
    """
    Make data frame with samples annotation,
    where all samples belong to the same group named ``"all_samp"``.

    Parameters
    ----------
    df_data : pandas.DataFrame
        Data frame with Index = samples.

    Returns
    -------
    pandas.DataFrame
        Data frame with the same Index and a single column
        ``"group"`` filled with a constant string ``"all_samp"``.
    """

    return pd.DataFrame({"group": "all_samp"}, index=df_data.index)


#######################################################
# Fast math functions for correlations and regression #
#######################################################


def compute_pearson_columns(X):
    """
    Compute Pearson's correlation for
    all-vs-all columns of input 2D array.

    Parameters
    ----------
    X : numpy.ndarray
        2D array with variables of interest over columns.

    Returns
    -------
    numpy.array
        1D array of correlations (diagonal 1s are not included).
        For speed-up purposes, data type is ``float32``.
    
    Notes
    -----
    The `numpy.triu_indices` function with parameter ``k=1`` can be used
    to establish correspondence between index of the resulting
    flat array and standard 2D correlation matrix.
    """

    corrs = np.corrcoef(X, rowvar=False, dtype="float32")
    corrs = flatten_corrs(corrs)
    return corrs


@njit(parallel=True)
def flatten_corrs(corrs_2d):
    """
    Convert 2D correlation matrix into 1D array.
    This is parallel version with result equivalent to
    ``corrs_2d[numpy.triu_indices(n_features, k=1)]``.

    Parameters
    ----------
    corrs_2d : numpy.ndarray
        Square ``(n, n)`` symmetric matrix with correlations.

    Returns
    -------
    numpy.array
        1D array with correlations (without diagonal 1s).
        Size of array is ``n * (n - 1) // 2``.
    """

    n_features = corrs_2d.shape[0]
    n_pairs = n_features * (n_features + 1) // 2
    # 1d correlations will not have diagonal 1s
    corrs_1d = np.zeros((n_pairs - n_features,), dtype="float32")

    for i in prange(n_features):
        for j in range(i + 1, n_features):
            # flat index of (i, j) in upper-triangular matrix without diagonal
            # This is reverse of np.triu_indices(n_features, k=1)
            k = j - 1 + i * (n_features - 1) - i * (i + 1) // 2
            corrs_1d[k] = corrs_2d[i, j]

    return corrs_1d


def compute_corr_pvalues(corrs, n):
    """
    Compute p-values testing null hypothesis
    of input Pearson's correlations equal to zero.

    Parameters
    ----------
    corrs : numpy.array
        1D array with correlations.
    n : int
        Sample size used when computing correlations.

    Returns
    -------
    numpy.array
        1D array of p-values corresponding to `corrs`.
    
    Notes
    -----
    This function produces the same p-values as the
    `scipy.stats.pearsonr` function with default parameters.
    """

    dist = beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)
    return 2*dist.cdf(-np.abs(corrs))


def regress_out_columns(X1, X2):
    """
    Regress out all columns of `X2` from each columns of `X1`.

    Parameters
    ----------
    X1 : numpy.ndarray
        2D array with variables of interest over columns.
    X2 : numpy.ndarray
        2D array with variables to residualize over columns.
        Should have the same number of rows and `X1`.

    Returns
    -------
    residuals : numpy.ndarray
        Residuals after regressing out columns of `X2` from
        each column of `X1`. Has the same shape as `X1`.

    rsquareds : numpy.array
        1D array of ``R^2`` values corresponding to regressions
        fit for each column of `X1`.
    """

    # Mean-center columns of X1 and X2 before regression,
    # so we can fit a model without intercept
    X1 = X1 - np.mean(X1, axis=0)
    X2 = X2 - np.mean(X2, axis=0)

    coef = np.linalg.lstsq(X2, X1, rcond=None)[0]
    residuals = X1 - X2 @ coef
    rsquareds = 1 - np.sum(residuals**2, axis=0) / (np.var(X1, axis=0) * X1.shape[0])

    return residuals, rsquareds


##################################################
# Other math (peak finding, hypergeometric test) #
##################################################


def find_peak(
    scores,
    patience=2,
    min_rel_improvement=0.05,
    max_rel_diff_from_max=0.5
):
    """
    Find peak with early stopping criteria.

    - `patience` rounds with no `min_rel_improvement` compared to
      the previous maximum value (plateau reached).
    - relative difference between global maximum and current maximum
      is at most `max_rel_diff_from_max` (not terminating too early).

    Parameters
    ----------
    scores : numpy.array
        1D array with scores used to find the peak.
    patience : int, optional, default=2
        See description above.
    min_rel_improvement : float, optional, default=0.05
        See description above.
    max_rel_diff_from_max : float, optional, default=0.5
        See description above.

    Returns
    -------
    int
        Index of peak element in the `scores` array.
    """

    global_max_score = np.max(scores)

    # Our metric is non-negative
    if global_max_score <= 0:
        return 0

    # Early stopping implementation
    peak_idx = 0
    iters_non_increasing = 0

    for idx in range(1, len(scores)):
        if scores[peak_idx] != 0:
            rel_improvement = (scores[idx] - scores[peak_idx]) / scores[peak_idx]
        else:
            rel_improvement = np.inf
        
        if rel_improvement >= min_rel_improvement:
            peak_idx = idx
            iters_non_increasing = 0
        else:
            iters_non_increasing += 1
        
        rel_diff_from_max = (global_max_score - scores[peak_idx]) / global_max_score

        if (
            iters_non_increasing >= patience and
            rel_diff_from_max <= max_rel_diff_from_max
        ):
            break

    return peak_idx


@njit(parallel=True)
def hypergeom_pvalues(M_arr, n_arr, N_arr, k_arr):
    """
    Parallel computation of hypergeometric test
    p-values over arrays of values ``M, n, N, k``.
    Semantics for argument names are the same as in
    `scipy.stats.hypergeom`.

    Parameters
    ----------
    M_arr : numpy.array
        Total number of objects in each test.
    n_arr : numpy.array
        Number of Type I objects in each test.
    N_arr : numpy.array
        Number of drawn objects in each test.
    k_arr : numpy.array
        Number of Type I drawn objects in each test.

    Returns
    -------
    numpy.array
        Array of p-values corresponding to each test.
        Minimum value of p-value is defined by
        ``numpy.finfo(numpy.float64).smallest_normal``.
    """

    global SMALLEST_FLOAT64

    pvalues = np.zeros(M_arr.shape, dtype="float64")
    for i in prange(M_arr.shape[0]):
        M, n, N, k = M_arr[i], n_arr[i], N_arr[i], k_arr[i]

        log_binom1 = log_binom_coef(n, k)
        log_binom2 = log_binom_coef(M - n, N - k)
        log_binom3 = log_binom_coef(M, N)
        # We do this explicitly on first iterations to avoid
        # division by zero in corner cases in for loop
        pvalue = np.exp(log_binom1 + log_binom2 - log_binom3)

        for k1 in range(k + 1, min(n, N) + 1):
            log_binom1 += np.log((n - k1 + 1) / k1)
            log_binom2 += np.log((N - k1 + 1) / (M - n - N + k1))
            pvalue += np.exp(log_binom1 + log_binom2 - log_binom3)
        
        # Sometimes pvalue could go beyond 1.0 a bit because of
        # numerical error reasons
        # Also, instead of p = 0, we report smallest "normal" positive float64
        pvalues[i] = max(min(pvalue, 1.0), SMALLEST_FLOAT64)

    return pvalues


@njit(parallel=False)
def log_binom_coef(n, k):
    """
    Logarithm of binomial coefficient ``(n k)``.

    Parameters
    ----------
    n, k : int
        Input numbers.

    Returns
    -------
    float
        Binomial coefficient.
    """

    k = min(k, n - k)
    res = 0.0
    for i in range(k):
        res += np.log((n - i) / (k - i))
    
    return res


def compute_aggregated_scores(df):
    """
    Compute aggregated statistics from the data frame
    of feature-wise scores.

    Parameters
    ----------
    df : pandas.DataFrame
        Data frame produced by CorrAdjust.compute_feature_scores.

    Returns
    -------
    agg_BP_at_K : float
        Aggregated balanced precision at K.
    agg_enrichment: float
        Aggregated enrichment.
    agg_pvalue: float
        Aggregated p-value.
    """

    df = df.dropna()
    agg_total_pos = df["shared_pairs@total"].str.split("/").str[0].astype("Int64").sum()
    agg_total = df["shared_pairs@total"].str.split("/").str[1].astype("Int64").sum()
    agg_K_pos = df["shared_pairs@K"].str.split("/").str[0].astype("Int64").sum()
    agg_K = df["shared_pairs@K"].str.split("/").str[1].astype("Int64").sum()

    # Balanced precision at K
    agg_BP_at_K_TP = agg_K_pos / agg_K * (agg_total - agg_total_pos)
    agg_BP_at_K_FP = (agg_K - agg_K_pos) / agg_K * agg_total_pos
    agg_BP_at_K = agg_BP_at_K_TP / (agg_BP_at_K_TP + agg_BP_at_K_FP)

    # Enrichment
    expected = agg_K * agg_total_pos / agg_total
    agg_enrichment = agg_K_pos / expected

    # P-value
    agg_pvalue = hypergeom_pvalues(
        np.array([agg_total]), np.array([agg_total_pos]),
        np.array([agg_K]), np.array([agg_K_pos])
    )[0]

    return agg_BP_at_K, agg_enrichment, agg_pvalue