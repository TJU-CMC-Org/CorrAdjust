import sys
import os
import parallel_sort

import pandas as pd
import numpy as np

import pyarrow as pa
from pyarrow import csv

from sklearn.decomposition import PCA
from kneed import KneeLocator
from tqdm import tqdm
from datetime import datetime

from corradjust.metrics import *
from corradjust.plots import *
from corradjust.utils import *


class CorrAdjust:
    """
    The main CorrAdjust class.

    Parameters
    ----------
    df_feature_ann : pandas.DataFrame
        A data frame with index and 2 columns 
        providing annotation for features.

        - Index: unique feature ids that are also used in the data table.
          For example, ENSEMBL gene ids or small RNA license plates.
        - Column ``feature_name``: user-friendly names of the features,
          which are also used in the reference ``.gmt`` files.
          For example, gene symbols or ``"hsa-miR-xx-5p"`` notation for miRNAs.
        - Column ``feature_type``: discrete set of feature types.
          E.g., if you are analyzing only mRNA-seq data, put ``"mRNA"`` for all features;
          if you are integrating miRNA, mRNA, or any other data type,
          you could use more than one type (e.g., ``"miRNA"`` and ``"mRNA"``).
          Feature types shouldn't contain ``"-"`` character.

    ref_feature_colls : dict
        A dictionary with information about reference feature set collections.
        Each collection is an item of `ref_feature_colls` with key being
        collection name (string) and value being another dict with the
        following structure.

        | {
        |     ``"path"`` : `str`
        |         Relative or absolute path to the ``.gmt`` file with reference feature sets.
        |     ``"sign"`` : ``{"positive", "negative", "absolute"}``
        |         Determines how to rank feature-feature correlations.
        |             ``"absolute"``: absolute values of correlations
                      (highest absolute correlations go first).
        |             ``"positive"``: descending sorting of correlations
                      (most positive correlations go first).
        |             ``"negative"``: ascending sorting of correlations
                      (most negative correlations go first).
        |     ``"feature_pair_types"`` : `list`
        |         List of strings showing types of correlations you are interested in.
                  For example, ``["mRNA-mRNA"]`` or ``["miRNA-mRNA"]``.
        |     ``"high_corr_frac"`` : float
        |         Fraction of all feature pairs with most highly ranked correlations
                  that will be used for scoring (alpha in the paper's notation).
        | }.
    
    out_dir : str
        Path to the directory where the output files will be stored.
        The directory will be created if it doesn't exist.
    winsorize : float or None, optional, default=0.05
        If `float`, specifies fraction of the lowest and the highest
        data values that will be winsorized.
        If ``None``, no winsorizing will be applied.
        Winsorizing is train/test-set friendly, i.e.,
        test data has no influence on training data.
    shuffle_feature_names : bool or list, optional, default=False
        If ``True``, ``feature_name`` column of `df_feature_ann` will be randomly permuted.
        If list of feature types, e.g., ``["mRNA"]``, permutation will be conducted
        only within the corresponding feature types.
        If ``False``, no permutations will be conducted.
    metric : {"enrichment-based", "BP@K"}, optional, default="enrichment-based"
        Metric for evaluating feature correlations (see CorrAdjust paper for details).
        Please use the default option unless you are absolutely sure
        why do you need balanced precision at K (BP@K).
    min_pairs_to_score : int, optional, default=1000
        Minimum number of total pairs containing a feature
        (N_j in the paper notation) to include the feature
        in the score computations. Inequaility must hold
        both in training and validation pair sets.
    random_seed : int, optional, default=17
        Random seed used by PCA and other non-deterministic procedures.
    title : str or None, optional, default=None
        If string, it will be shown in top-left corner of all generated plots.
    verbose : bool, optional, default=True
        Whether to print progress messages to stderr.
    
    Attributes
    ----------
    confounder_PCs : list
        The list of confounder PC indices (0-based).
        Populated by calling `fit` method.
    PCA_model : sklearn.decomposition.PCA
        PCA instance fit on training data. 
        Populated by calling `fit` and `fit_PCA` methods.
    winsorizer : Winsorizer
        Winsorizer instance fit on training data. 
        Populated by calling `fit` and `fit_PCA` methods.
    mean_centerizer : MeanCenterizer
        MeanCenterizer instance fit on training data. 
        Populated by calling `fit` and `fit_PCA` methods.
    corr_scorer : CorrScorer
        Instance of CorrScorer.
        Populated by constructor.
    out_dir : str
        Output directory.
        Populated by constructor.
    title : str
        Title added to left-top corner of the plots.
        Populated by constructor.
    metric : {"enrichment-based", "BP@K"}
        Metric for evaluating feature correlations.
        Populated by constructor.
    random_seed : int
        Random seed.
        Populated by constructor.
    verbose : bool
        Whether to print progress messages.
        Populated by constructor.
    """

    def __init__(
        self,
        df_feature_ann,
        ref_feature_colls,
        out_dir,
        winsorize=0.05,
        shuffle_feature_names=False,
        metric="enrichment-based",
        min_pairs_to_score=1000,
        random_seed=17,
        title=None,
        verbose=True
    ):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        if winsorize is not None:
            assert (
                isinstance(winsorize, float) and 0 < winsorize < 1
            ), "winsorize must be None or float between 0 and 1."
            self.winsorizer = Winsorizer(alpha=winsorize)
        else:
            self.winsorizer = None
        
        assert title is None or isinstance(title, str), "title must be None or str."
        self.title = title
        
        # Load reference data in a data structure
        self.corr_scorer = CorrScorer(
            df_feature_ann,
            ref_feature_colls,
            shuffle_feature_names,
            metric,
            min_pairs_to_score,
            random_seed,
            verbose
        )
        self.metric = metric
        self.random_seed = random_seed
        self.verbose = verbose
    

    def fit(
        self,
        df_data,
        df_samp_ann=None,
        method="greedy",
        n_PCs=None,
        n_iters=None
    ):
        """
        Find optimal set of PCs to regress out.
        Calling this method sets `confounder_PCs` attribute.

        It also generates several files in `out_dir`:

        - ``fit.tsv`` : file with the scores across the optimization trajectory.
        - ``fit.{sample_group}.png`` : visualization of ``fit.tsv`` for each sample group.
        - ``fit.log`` : detailed log file updated at each iteration.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Training data. Index = samples, columns = feature ids.
            Data should be normalized so that feature values
            are comparable between samples.
            Feature ids should perfectly match the ones used in
            `df_feature_ann` while initializing the object.
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame with index and one column
            providing group annotation of samples.

            - Index: sample names (should match index of `df_data`).
            - Column ``group``: discrete set of group names (e.g., ``"normal"`` and ``"tumor"``).
              Group name cannot be ``"mean"``.

            If `df_samp_ann` is ``None``, the method automatically generates it
            with one sample group called ``all_samp``.
        method : {"greedy", "first-n"}, optional, default="greedy"
            Optimization method.

            - ``"greedy"`` (recommended default): searches for the best PC
              to remove on each iteration (quadratic complexity).
            - ``"first-n"``: iteratively removes PCs based on
              % variance they explain (linear complexity). This is
              the concept used by `sva_network` function of the `sva` package.
        n_PCs : int or None, optional, default=None
            Number of PCs to choose from during the optimization.
            By default (``None``), number of PCs is estimated as a knee of 
            % cumulative variance plot.
        n_iters : int or None, optional, default=None
            How many optimization iterations to perform.
            If ``None``, `n_iters` is set to `n_PCs`.
        """
        
        self._check_df_data(df_data)
        self._check_df_samp_ann(df_samp_ann, df_data)
        assert (
            method in {"greedy", "first-n"}
        ), f"Method should be 'greedy' or 'first-n' but '{method}' found."
        assert n_PCs is None or isinstance(n_PCs, int), "n_PCs must be None or int."
        assert n_iters is None or isinstance(n_iters, int), "n_iters must be None or int."
        
        # If there are no sample groups, make a dummy one
        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)

        # Mean-centering and winsorization occurs in .fit_PCA,
        # that's why it returns df_data
        PCs, PC_coefs, n_PCs_knee, df_data = self.fit_PCA(df_data, df_samp_ann)

        # If not explicitly passed, we use knee estimate for number of PCs
        if not n_PCs:
            n_PCs = n_PCs_knee
        else:
            # If user set more PCs than possible, we auto-correct that
            if n_PCs > PCs.shape[1]:
                print(
                    f"WARNING | You set {n_PCs} PCs, but the data has only "
                    f"{PCs.shape[1]} PCs. The run will continue with "
                    f"{PCs.shape[1]} PCs.",
                    file=sys.stderr
                )
                n_PCs = PCs.shape[1]
        
        PCs = PCs[:, :n_PCs]
        PC_coefs = PC_coefs[:n_PCs, :]

        if n_iters is None:
            n_iters = n_PCs

        self._search_confounder_PCs(
            df_data, df_samp_ann,
            PCs, PC_coefs,
            method, n_iters
        )


    def fit_PCA(self, df_data, df_samp_ann=None, plot_var=False):
        """
        Compute principal components of training data.
        The data are centered and optionally winsorized before
        computing PCA in train/test-friendly way.
        
        Calling this method sets `PCA_model` attribute.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Training data. Index = samples, columns = feature ids.
            See `fit` method.
        df_samp_ann : pandas.DataFrame, optional, default=None
            A data frame providing group annotation of samples.
            See `fit` method.
        plot_var : bool, optional, default=False
            Whether to make plots with % variance explained.
            The plots will be saved as ``"PCA_var.png"`` in `out_dir`.

        Returns
        -------
        PCs : numpy.ndarray
            Array of principal components with shape ``(n_samples, n_components)``.
        PC_coefs : numpy.ndarray
            Array of PC coefficients with shape ``(n_components, n_features)``.
        n_PCs_knee : int
            Number of PCs estimated using knee (a.k.a. elbow) method.
        df_data : pandas.DataFrame
            Centered and (optionally) winsorized training data.
        """
        
        self._check_df_data(df_data)
        self._check_df_samp_ann(df_samp_ann, df_data)
        assert plot_var in {True, False}, "plot_var must True or False."

        if self.verbose:
            print(f"{datetime.now()} | Computing PCA...", file=sys.stderr)
        
        # If there are no sample groups, make a dummy one
        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)
        
        # fit_PCA is always called from fit, but might also be used standalone
        # That's why we decided to center and winsorize data in this method.
        if self.winsorizer is not None:
            self.winsorizer.fit(df_data)
            df_data = self.winsorizer.transform(df_data)

        self.mean_centerizer = MeanCenterizer()
        self.mean_centerizer.fit(df_data, df_samp_ann)
        df_data = self.mean_centerizer.transform(df_data, df_samp_ann)

        # Initialize and fit PCA
        self.PCA_model = PCA(random_state=self.random_seed)
        self.PCA_model.fit(df_data)
        PCs = self.PCA_model.transform(df_data)
        PC_coefs = self.PCA_model.components_
        
        # Find number of PCs with knee method
        kneedle = KneeLocator(
            np.arange(1, len(self.PCA_model.explained_variance_ratio_) + 1),
            np.cumsum(self.PCA_model.explained_variance_ratio_),
            S=1.0, curve="concave", direction="increasing"
        )
        n_PCs_knee = kneedle.knee

        if plot_var: 
            plotter = PCAVariancePlotter()
            plotter.plot(self.PCA_model, n_PCs_knee)
            plotter.save_plot(f"{self.out_dir}/PCA_var.png", title=self.title)

        return PCs, PC_coefs, n_PCs_knee, df_data
    
    
    def transform(self, df_data, df_samp_ann=None):
        """
        Residualize identified confounder PCs from input data.
        This method should be called after `fit`, or after manually
        calling `fit_PCA` and setting `confounder_PCs` attribute.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            Feature ids should perfectly match the ones used in
            `df_feature_ann` while initializing the object.
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame providing group annotation of samples.
            This argument is mandatory if it was used during `fit` call.

        Returns
        -------
        df_data_clean : pandas.DataFrame
            Cleaned data.
        df_rsquareds : pandas.DataFrame
            Data frame with ``R^2`` values for each feature.
        """
        
        assert (
            hasattr(self, "confounder_PCs")
        ), (
            "CorrAdjust instance must have confounder_PCs attribute when calling transform. "
            "Call fit method or manually set the attribute."
        )
        assert (
            hasattr(self, "PCA_model") and hasattr(self, "mean_centerizer")
        ), "fit_PCA should be called prior to transform."
        
        self._check_df_data(df_data)
        self._check_df_samp_ann(df_samp_ann, df_data, after_training=True)
        
        if self.winsorizer is not None:
            df_data = self.winsorizer.transform(df_data)
        df_data = self.mean_centerizer.transform(df_data, df_samp_ann)

        X_raw = df_data.to_numpy()
        X_PCA = self.PCA_model.transform(df_data)
        
        # Regress out optimal PCs from data
        X_clean, rsquareds = regress_out_columns(
            X_raw, X_PCA[:, self.confounder_PCs]
        )

        df_data_clean = pd.DataFrame(X_clean, index=df_data.index, columns=df_data.columns)
        df_rsquareds = pd.DataFrame(rsquareds, index=df_data.columns, columns=["rsquared"])
        
        return df_data_clean, df_rsquareds
    
    
    def compute_feature_scores(
        self,
        df_data,
        df_samp_ann=None,
        samp_group=None,
        pairs_subset="all"
    ):
        """
        Compute enrichment scores and p-values
        for feature-feature correlations in given data.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            Feature ids should perfectly match the ones used in
            `df_feature_ann` while initializing the object.
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame providing group annotation of samples.
            This argument is mandatory if it was used during `fit` call.
        samp_group : str or None, default=None
            If `df_samp_ann` is passed, this argument is mandatory
            and should refer to one of the sample groups in `df_samp_ann`.
        pairs_subset : {"all", "training", "validation"}, optional, default="all"
            Which set of feature pairs to use for computing scores.

        Returns
        -------
        dict
            Dict with data frames containing feature-wise
            enrichment scores and p-values.
            It has the following structure:

            | {
            |     ``"Clean"`` : {
            |         ``ref_feature_coll`` : pandas.DataFrame
            |     },
            |     ``"Raw"``: {...}
            | }.
        """
        
        assert (
            hasattr(self, "PCA_model") and hasattr(self, "mean_centerizer")
        ), "fit_PCA should be called prior to compute_feature_scores."
        
        self._check_df_data(df_data)
        
        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)
            samp_group = df_samp_ann.iloc[0]["group"]
        
        self._check_df_samp_ann(df_samp_ann, df_data, after_training=True)
        assert samp_group in set(df_samp_ann["group"]), f"{samp_group} is not present in sample annotation."

        # Subset is training/validation/all
        subset_idxs = {"training": 0, "validation": 1, "all": 2}
        assert (
            pairs_subset in subset_idxs
        ), f"pairs_subset must be one of 'all', 'training', 'validation', but '{pairs_subset}' was found."
        
        feature_scores = {}
        corrs_raw, corrs_clean = self._compute_raw_and_clean_corrs(df_data, df_samp_ann, samp_group)

        corr_scores_raw = self.corr_scorer.compute_corr_scores(corrs_raw, full_output=True)
        if hasattr(self, "confounder_PCs"):
            corr_scores_clean = self.corr_scorer.compute_corr_scores(corrs_clean, full_output=True)
        else:
            corr_scores_clean = None
        
        for state, corr_scores in [("Raw", corr_scores_raw), ("Clean", corr_scores_clean)]:
            if corr_scores is None:
                feature_scores[state] = None
                continue
            else:
                feature_scores[state] = {}

            for ref_feature_coll in corr_scores:
                subset_idx = subset_idxs[pairs_subset]
                total_pos = self.corr_scorer.data[ref_feature_coll]["features_total_pos"][subset_idx]
                total_neg = self.corr_scorer.data[ref_feature_coll]["features_total_neg"][subset_idx]

                df = pd.DataFrame({
                    "feature_id": self.corr_scorer.feature_ids,
                    "feature_name": self.corr_scorer.feature_names,
                    "ref_pairs@K": corr_scores[ref_feature_coll][f"TPs_at_K {pairs_subset}"],
                    "K": corr_scores[ref_feature_coll][f"num_pairs {pairs_subset}"],
                    "ref_pairs@total": total_pos,
                    "total": total_pos + total_neg,
                    "enrichment": corr_scores[ref_feature_coll][f"enrichments {pairs_subset}"],
                    "balanced_precision": corr_scores[ref_feature_coll][f"BPs_at_K {pairs_subset}"],
                    "pvalue": corr_scores[ref_feature_coll][f"pvalues {pairs_subset}"],
                    "padj": corr_scores[ref_feature_coll][f"pvalues_adj {pairs_subset}"],
                })
                df = df.set_index("feature_id")

                df["ref_pairs@K"] = df["ref_pairs@K"].astype("string") + "/" + df["K"].astype("string")
                df["ref_pairs@total"] = df["ref_pairs@total"].astype("string") + "/" + df["total"].astype("string")
                df = df.drop(columns=["K", "total"])

                df = df.sort_values(["padj", "pvalue", "feature_id"])
                feature_scores[state][ref_feature_coll] = df

        return feature_scores
            

    def make_volcano_plot(self,
        feature_scores,
        plot_filename,
        **plot_kwargs
    ):
        """
        Make volcano plot.

        Parameters
        ----------
        feature_scores : dict
            Dict produced by `compute_feature_scores` method.
        plot_filename : str
            Path to the figure (with extension, e.g., ``.png``).
        **plot_kwargs
            Other keyword arguments controlling plot
            aesthetics passed to `VolcanoPlotter`.

        Returns
        -------
        VolcanoPlotter
            Instance of `VolcanoPlotter`. At this point,
            the plot is already saved in the file, so you
            can change it through fig/ax and re-save.
        """

        plotter = VolcanoPlotter(self.corr_scorer, **plot_kwargs)
        plotter.plot(feature_scores)
        plotter.save_plot(f"{self.out_dir}/{plot_filename}", title=self.title)

        return plotter
    

    def make_corr_distr_plot(
        self,
        df_data,
        plot_filename,
        df_samp_ann=None,
        samp_group=None,
        pairs_subset="all"
    ):
        """
        Visualize distribution of feature-feature correlations
        before and after PC correction.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            Feature ids should perfectly match the ones used in
            `df_feature_ann` while initializing the object.
        plot_filename : str
            Path to the figure (with extension, e.g., ``.png``).
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame providing group annotation of samples.
            This argument is mandatory if it was used during `fit` call.
        samp_group : str or None, default=None
            If `df_samp_ann` is passed, this argument is mandatory
            and should refer to one of the sample groups in `df_samp_ann`.
        pairs_subset : {"all", "training", "validation"}, optional, default="all"
            Which set of feature pairs to use for computing scores.

        Returns
        -------
        CorrDistrPlotter
            Instance of `CorrDistrPlotter`. At this point,
            the plot is already saved in the file, so you
            can change it through fig/ax and re-save.
        """
        
        assert (
            hasattr(self, "PCA_model") and hasattr(self, "mean_centerizer")
        ), "fit_PCA should be called prior to make_corr_distr_plot."
        
        self._check_df_data(df_data)
        
        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)
            samp_group = df_samp_ann.iloc[0]["group"]
        
        self._check_df_samp_ann(df_samp_ann, df_data, after_training=True)
        assert samp_group in set(df_samp_ann["group"]), f"{samp_group} is not present in sample annotation."

        # Subset is training/validation/all
        subset_idxs = {"training": 0, "validation": 1, "all": 2}
        assert (
            pairs_subset in subset_idxs
        ), f"pairs_subset must be one of 'all', 'training', 'validation', but '{pairs_subset}' was found."
        
        corrs_raw, corrs_clean = self._compute_raw_and_clean_corrs(df_data, df_samp_ann, samp_group)
        
        corr_scores_raw = self.corr_scorer.compute_corr_scores(corrs_raw, full_output=True)
        if hasattr(self, "confounder_PCs"):
            corr_scores_clean = self.corr_scorer.compute_corr_scores(corrs_clean, full_output=True)
        else:
            corr_scores_clean = None

        plotter = CorrDistrPlotter(self.corr_scorer, pairs_subset=pairs_subset)

        plotter.add_plots(corr_scores_raw, state="Raw")
        if hasattr(self, "confounder_PCs") and len(self.confounder_PCs):
            plotter.add_plots(corr_scores_clean, state="Clean")
        
        plotter.save_plot(f"{self.out_dir}/{plot_filename}", title=self.title)

        return plotter


    def export_corrs(
        self,
        df_data,
        filename,
        df_samp_ann=None,
        samp_group=None,
        chunk_size=1000000
    ):
        """
        Export feature-feature correlations to file.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            Feature ids should perfectly match the ones used in
            `df_feature_ann` while initializing the object.
        filename : str
            Path to the export table with ``.tsv`` extension.
        df_samp_ann : pandas.DataFrame or None, optional, default=None
            A data frame providing group annotation of samples.
            This argument is mandatory if it was used during `fit` call.
        samp_group : str or None, default=None
            If `df_samp_ann` is passed, this argument is mandatory
            and should refer to one of the sample groups in `df_samp_ann`.
        chunk_size : int, optional, default=1000000
            How many rows are exported in a single chunk.
        """
        
        assert (
            hasattr(self, "PCA_model") and hasattr(self, "mean_centerizer")
        ), "fit_PCA should be called prior to export_corrs."
        
        self._check_df_data(df_data)
        
        if df_samp_ann is None:
            df_samp_ann = make_dummy_samp_ann(df_data)
            samp_group = df_samp_ann.iloc[0]["group"]
        
        self._check_df_samp_ann(df_samp_ann, df_data, after_training=True)
        assert samp_group in set(df_samp_ann["group"]), f"{samp_group} is not present in sample annotation."

        assert isinstance(chunk_size, int), "chunk_size must be int."        

        corrs_raw, corrs_clean = self._compute_raw_and_clean_corrs(df_data, df_samp_ann, samp_group)
        
        if hasattr(self, "confounder_PCs"):
            corrs_sorted_args = parallel_sort.argsort(corrs_clean)
        else:
            corrs_sorted_args = parallel_sort.argsort(corrs_raw)

        n_samples, n_features = df_data.shape
        triu_rows, triu_cols = np.triu_indices(n_features, k=1)
        
        if self.verbose:
            print(f"{datetime.now()} | Starting export to file...", file=sys.stderr)        

        # We explicitly write header to csv file
        # because pyarrow 
        out_file = open(f"{self.out_dir}/{filename}", "w")
        header = [
            "feature_id1", "feature_id2",
            "feature_name1", "feature_name2",
            "feature_type1", "feature_type2",
            "corr_clean", "pvalue_clean",
            "corr_raw", "pvalue_raw",
        ] 
        ref_feature_coll_header = [
            f"{ref_feature_coll}_{col_type}"
            for ref_feature_coll in self.corr_scorer.data.keys()
            for col_type in ["flag", "trainval"]
        ]
        header += ref_feature_coll_header
        header = "\t".join(header)
        out_file.write(f"{header}\n")
        out_file.close()
        
        n_chunks = corrs_sorted_args.shape[0] // chunk_size
        n_chunks += 1 if corrs_sorted_args.shape[0] % chunk_size else 0

        if self.verbose:
            chunk_iterator = tqdm(range(n_chunks))
        else:
            chunk_iterator = range(n_chunks)
        
        for chunk_idx in chunk_iterator:
            start = chunk_idx * chunk_size
            end = start + chunk_size
            idxs = corrs_sorted_args[start:end]

            pvalues_raw = compute_corr_pvalues(corrs_raw[idxs], n_samples)
            if hasattr(self, "confounder_PCs"):
                if len(self.confounder_PCs):
                    pvalues_clean = compute_corr_pvalues(corrs_clean[idxs], n_samples - len(self.confounder_PCs))
                else:
                    pvalues_clean = pvalues_raw
            else:
                pvalues_clean = np.zeros(pvalues_raw.shape)
                pvalues_clean[:] = np.nan

            regular_columns = {
                "feature_id1": self.corr_scorer.feature_ids[triu_rows[idxs]],
                "feature_id2": self.corr_scorer.feature_ids[triu_cols[idxs]],
                "feature_name1": self.corr_scorer.feature_names[triu_rows[idxs]],
                "feature_name2": self.corr_scorer.feature_names[triu_cols[idxs]],
                "feature_type1": self.corr_scorer.feature_types[triu_rows[idxs]],
                "feature_type2": self.corr_scorer.feature_types[triu_cols[idxs]],
                "corr_clean": corrs_clean[idxs],
                "pvalue_clean": pvalues_clean,
                "corr_raw": corrs_raw[idxs],
                "pvalue_raw": pvalues_raw
            }

            ref_feature_columns = {}
            for ref_feature_coll in self.corr_scorer.data:
                ref_feature_columns[f"{ref_feature_coll}_flag"] = (
                    self.corr_scorer.data[ref_feature_coll]["mask"][idxs]
                )
                ref_feature_columns[f"{ref_feature_coll}_trainval"] = (
                    self.corr_scorer.data[ref_feature_coll]["train_val_mask"][idxs]
                )

            tab = pa.table({
                **regular_columns,
                **ref_feature_columns
            })

            if chunk_idx == 0:
                # This cannot be done before the loop because we need to know tab.schema
                write_options = csv.WriteOptions(include_header=False, delimiter="\t", quoting_style="none")
                out_file = open(f"{self.out_dir}/{filename}", "ab")
                writer = csv.CSVWriter(out_file, tab.schema, write_options=write_options)
            
            writer.write(tab)
        
        writer.close()
        out_file.close()
    

    def _search_confounder_PCs(
        self,
        df_data,
        df_samp_ann,
        PCs,
        PC_coefs,
        method,
        n_iters
    ):
        """
        Iteratively residualize PCs from training data
        to maximize number of reference pairs among
        highly correlated feature pairs.

        Parental `fit` method describes files generated by this method.

        Parameters
        ----------
        df_data : pandas.DataFrame
            See `fit` method.
        df_samp_ann : pandas.DataFrame or None
            See `fit` method.
        PCs : numpy.ndarray
            2D array of PCs (returned by `fit_PCA`).
        PC_coefs : numpy.ndarray
            2D array of PC coefficients (returned by `fit_PCA`).
        method : str
            See `fit` method.
        n_iters : _type_
            See `fit` method.
        """

        n_PCs = PCs.shape[1]
        X = df_data.to_numpy()

        used_PCs = []
        unused_PCs = list(range(n_PCs))
        
        if self.verbose:
            print(
                f"{datetime.now()} | Starting PC optimization...",
                file=sys.stderr
            )
            # +1 stands for the iteration before cleaning
            if method == "greedy":
                total_iterations = sum([n_PCs - i for i in range(n_iters)]) + 1
            else:
                total_iterations = n_iters + 1
            prog_bar = tqdm(total=total_iterations)
        else:
            prog_bar = None

        log_file = open(f"{self.out_dir}/fit.log", "w")
        
        # Compute scores with raw data
        print(
            f"{datetime.now()} | Starting iteration #0 (raw correlations)...\n",
            file=log_file, flush=True
        )
        iter_scores, main_score = self._compute_corrs_and_scores(
            X, df_samp_ann, log_file, prog_bar, "Scores before correction"
        )

        # This list contains values of best scores across
        # algorithm iterations. First values are the scores
        # before regressing out any PCs
        best_iter_scores = [iter_scores]
                
        for _ in range(n_iters):
            print(
                f"\n{datetime.now()} | Starting iteration #{len(used_PCs) + 1}...\n",
                file=log_file, flush=True
            )

            main_scores = []
            full_scores = []
            for PC in unused_PCs:
                # Regress out PC
                X_resid = X - PCs[:, [PC]] @ PC_coefs[[PC], :]
                
                log_message = f"Trying PC {PC}" if method == "greedy" else None
                iter_scores, main_score = self._compute_corrs_and_scores(
                    X_resid, df_samp_ann, log_file, prog_bar, log_message
                )
                
                main_scores.append(main_score)
                full_scores.append(iter_scores)

                # If we break the loop here, then
                # the chosen PCs will be the one with the lowest index 
                if method == "first-n":
                    break

            PC_max_idx = np.argmax(main_scores)
            PC_max = unused_PCs[PC_max_idx]
            main_score_max = main_scores[PC_max_idx]
            full_score_max = full_scores[PC_max_idx]
            used_PCs.append(PC_max)
            unused_PCs.remove(PC_max)
            best_iter_scores.append(full_score_max)

            print(
                f"\n{datetime.now()} | PC {PC_max} selected on iteration #{len(used_PCs)}; "
                f"Main score: {main_score_max}; "
                f"All scores: {full_score_max}",
                file=log_file, flush=True
            )

            # Update data matrix with PC_max
            X = X - PCs[:, [PC_max]] @ PC_coefs[[PC_max], :]

        log_file.close()
        if self.verbose:
            prog_bar.close()

        # Convert scores into data frame
        df_best_iter_scores = self._best_iter_scores_to_df(best_iter_scores, used_PCs)
        df_best_iter_scores.to_csv(f"{self.out_dir}/fit.tsv", sep="\t")

        # Find the peak for number of iterations using early stopping
        # Mean of all sample groups, mean of reference collections
        main_score_col = "mean;mean;validation"        
        peak_iter = find_peak(df_best_iter_scores[main_score_col].to_numpy())
        self.confounder_PCs = df_best_iter_scores["PC"][1:1 + peak_iter].to_list()

        # Make line plot for each sample group
        self._make_best_iter_scores_plot(df_best_iter_scores, peak_iter, df_samp_ann)


    def _compute_corrs_and_scores(
        self,
        X,
        df_samp_ann,
        log_file,
        prog_bar,
        log_message
    ):
        """
        Conduct one iteration of PC optimization
        by computing and scoring correlations.
        Scores will be computed separately
        for each sample group, mean scores will be also returned.

        Parameters
        ----------
        X : numpy.ndarray
            Training data, rows = samples, columns = features.
        df_samp_ann : pandas.DataFrame
            See `fit` method.
        log_file : file
            Writing-open log file.
        prog_bar : tqdm.std.tqdm or None
            Progress bar (will be updated) or ``None``.
        log_message : str or None
            Title of the log message. If ``None``, nothing
            will be printed to log.

        Returns
        -------
        iter_scores : dict
            A dict of the following structure:

            | {
            |     ``sample_group_name`` : {
            |         ``ref_feature_coll`` : {
            |             ``"score training"`` : float,
            |             ``"score validation"`` : float,
            |             ``"score all"`` : float,
            |         }
            |     }
            | }.

        main_score : float
            Mean of training scores across all sample groups
            and reference collections, i.e.,
            ``iter_scores["mean"]["mean"]["score training"]``.
        """

        metric_subsets = ["training", "validation", "all"]
        iter_scores = {}

        # Compute correlations and scores separately for each sample group
        unique_samp_groups = df_samp_ann["group"].unique()
        for samp_group in unique_samp_groups:
            samp_group_mask = np.where(df_samp_ann["group"] == samp_group)[0]

            # Compute correlations and scores
            corrs = compute_pearson_columns(X[samp_group_mask])
            corr_scores = self.corr_scorer.compute_corr_scores(corrs)

            # Re-arrange scores in a dict
            iter_scores[samp_group] = {
                ref_feature_coll: {
                    f"score {subset}": metrics[f"score {subset}"]
                    for subset in metric_subsets
                }
                for ref_feature_coll, metrics in corr_scores.items()
            }
            # Compute mean score across reference collections (within sample group)
            iter_scores[samp_group]["mean"] = {
                f"score {subset}": np.mean([
                    metrics[f"score {subset}"]
                    for ref_feature_coll, metrics in iter_scores[samp_group].items()
                ])
                for subset in metric_subsets
            }

        # We also compute mean across sample groups
        iter_scores["mean"] = {
            ref_feature_coll: {
                f"score {subset}": np.mean([
                    iter_scores[samp_group][ref_feature_coll][f"score {subset}"]
                    for samp_group in unique_samp_groups
                ])
                for subset in metric_subsets
            }
            for ref_feature_coll in iter_scores[unique_samp_groups[0]].keys()
        }
        main_score = iter_scores["mean"]["mean"]["score training"]

        if prog_bar is not None:
            prog_bar.update()

        if log_message is not None: 
            print(
                f"{datetime.now()} | {log_message}; "
                f"Main score: {main_score}; "
                f"All scores: {iter_scores}",
                file=log_file, flush=True
            )

        return iter_scores, main_score


    def _best_iter_scores_to_df(self, best_iter_scores, used_PCs):
        """
        Convert dict of scores into data frame.

        Parameters
        ----------
        best_iter_scores : list
            List of dicts with scores corresponding
            to the iterations of optimization. Each element
            of list should have a format specified in the return
            dict of the `_compute_corrs_and_scores` method.
        used_PCs : list
            List of PC indices corresponding to `best_iter_scores`.

        Returns
        -------
        pandas.DataFrame
            Pandas data frame with scores in columns and iterations on rows.

            - Index: iteration number. Starts with 0 (raw dataset before correction).
            - Column ``PC``: index of PC being regressed out on each iteration.
              First row always have NaN (no correction).
            - Score columns: these columns have the following format:
              ``"{sample_group_name};{ref_feature_coll};{training/validation/all}"``.
        """

        best_iter_scores = [
            {
                f"{samp_group};{ref_feature_coll};{metric.replace('score ', '')}": value
                for samp_group in iter_scores_dict
                for ref_feature_coll, metrics in iter_scores_dict[samp_group].items()
                for metric, value in metrics.items()
            }
            for iter_scores_dict in best_iter_scores
        ]
        df_best_iter_scores = pd.DataFrame.from_dict(best_iter_scores)
        df_best_iter_scores.index.name = "Iteration"
        df_best_iter_scores.insert(0, "PC", [np.nan] + used_PCs)
        df_best_iter_scores["PC"] = df_best_iter_scores["PC"].astype("Int64")

        return df_best_iter_scores


    def _make_best_iter_scores_plot(self, df_best_iter_scores, peak_iter, df_samp_ann):
        """
        Make optimization score line plots.
        If there is one sample group, only one plot
        with ``"fit.png"`` name will be produced.
        If there are multiple groups, one plot per group
        will be made: ``"fit.{group_name}.png"``, including
        group with mean values.

        Parameters
        ----------
        df_best_iter_scores : pandas.DataFrame
            Data frame generated using `_best_iter_scores_to_df` method.
        peak_iter : int
            This number is used to draw vertical dashed red
            line at the selected early stopping iteration.
        df_samp_ann : pandas.DataFrame
            See `fit` method.
        """

        unique_samp_groups = df_samp_ann["group"].unique()
        # If there is only one sample group, we don't plot "mean"
        if len(unique_samp_groups) > 1:
            unique_samp_groups = unique_samp_groups.tolist() + ["mean"]
        
        for samp_group in unique_samp_groups:
            # Leave only necessary columns
            df_best_iter_scores_group = df_best_iter_scores[[
                col for col in df_best_iter_scores.columns if col.startswith(samp_group)
            ]]
            
            plotter = GreedyOptimizationPlotter(
                samp_group_name=samp_group,
                metric=self.metric
            )
            plotter.plot(df_best_iter_scores_group, peak_iter)
            plotter.save_plot(f"{self.out_dir}/fit.{samp_group}.png", title=self.title)


    def _compute_raw_and_clean_corrs(self, df_data, df_samp_ann, samp_group):
        """
        Compute correlations before and after residualizing
        PC confounders for a given group of samples.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            See `transform` for details.
        df_samp_ann : pandas.DataFrame
            A data frame providing group annotation of samples.
            See `transform` for details.
        samp_group : str
            One of the sample groups in `df_samp_ann`.

        Returns
        -------
        corrs_raw : numpy.array
            1D array of correlations (diagonal 1s are not included).
        corrs_clean : numpy.array
            Same but with cleaned correlations.
        """
        
        # Preprocess the data
        df_data_orig = df_data.copy()
        if self.winsorizer is not None:
            df_data = self.winsorizer.transform(df_data)
        df_data = self.mean_centerizer.transform(df_data, df_samp_ann)

        # All samples are cleaned together 
        if hasattr(self, "confounder_PCs"):
            if len(self.confounder_PCs):
                # We feed original data, so it's not centered/winsorized twice
                df_data_clean, df_rsquareds = self.transform(df_data_orig, df_samp_ann)
            else:
                df_data_clean = df_data
        
        # Only samp_group samples are used to compute correlation
        if self.verbose:
            print(f"{datetime.now()} | Computing raw correlations for {samp_group}...", file=sys.stderr)
        
        X_raw = df_data.loc[df_samp_ann["group"] == samp_group].to_numpy()
        corrs_raw = compute_pearson_columns(X_raw)
        
        if hasattr(self, "confounder_PCs"):
            if len(self.confounder_PCs):
                if self.verbose:
                    print(f"{datetime.now()} | Computing corrected correlations for {samp_group}...", file=sys.stderr)

                X_clean = df_data_clean.loc[df_samp_ann["group"] == samp_group].to_numpy()
                corrs_clean = compute_pearson_columns(X_clean) 
            else:
                # If there are no PCs to clean, then we don't need to re-compute anything
                corrs_clean = corrs_raw
        else:
            corrs_clean = np.zeros(corrs_raw.shape)
            corrs_clean[:] = np.nan

        return corrs_raw, corrs_clean


    def _check_df_data(self, df_data):
        """
        Check validity of input data.

        Parameters
        ----------
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            See `fit` method.
        
        Raises
        ------
        AssertionError
            Several possible reasons:

            - Feature ids do not match ones in feature annotation.
            - There is a gene with constant expression.
        """

        assert (
            list(df_data.columns) == list(self.corr_scorer.feature_ids)
        ), "Columns of input data should match index of feature annotation."
        
        assert (
            np.all(df_data.std(axis=0) > 0)
        ), "Input data contains features with constant expression."
    

    def _check_df_samp_ann(self, df_samp_ann, df_data, after_training=False):
        """
        Check validity of sample annotation.

        Parameters
        ----------
        df_samp_ann : pandas.DataFrame or None
            Sample annotation data.
            See `fit` method.
        df_data : pandas.DataFrame
            Input data. Index = samples, columns = feature ids.
            See `fit` method.
        after_training : bool
            Whether this method is called after training.
        
        Raises
        ------
        AssertionError
            Several possible reasons:

            - Sample names in `df_samp_ann` and `df_data` do not match.
            - `df_samp_ann` lacks ``group`` column.
            - If ``after_training == True``, method will also check
              that group names of `df_samp_ann` match those in `mean_centerizer`.
        """

        if df_samp_ann is None:
            return

        assert (
            list(df_samp_ann.index) == list(df_data.index)
        ), "Index of input data should match index of sample annotation."
        
        assert (
            "group" in set(df_samp_ann.columns)
        ), "Column 'group' is not found in sample annotation."
        
        assert (
            "mean" not in set(df_samp_ann["group"])
        ), "Sample group cannot be named 'mean'."

        if after_training:
            assert (
                set(df_samp_ann["group"]) <= set(self.mean_centerizer.means.keys())
            ), "Sample annotation has some groups that were absent in training samples."