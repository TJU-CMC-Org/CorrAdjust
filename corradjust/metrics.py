import sys
import os
import io
import parallel_sort

import numpy as np

from datetime import datetime
from numba import njit, prange
from scipy.stats import false_discovery_control

from corradjust.utils import *


class CorrScorer:
    """
    Class that contains data structures
    for features and reference feature sets, and computes
    different scores given feature-feature correlations.
    See documentation of CorrAdjust class constructor
    for detailed description of constructor parameters (they are the same).

    Parameters
    ----------
    df_feature_ann : pandas.DataFrame with index and 2 columns
        A data frame providing annotation of features.
    ref_feature_colls : dict
        A dictionary with information about reference feature set collections.
    shuffle_feature_names : bool or list, optional, default=False
        Whether (and how) to permute feature names.
    metric : {"enrichment-based", "BP@K"}, optional, default="enrichment-based"
        Metric for evaluating feature correlations.
    min_pairs_to_score : int, optional, default=1000
        Minimum number of total pairs containing a feature
        to include the feature in the enrichment score computations.
    random_seed : int, optional, default=17
        Random seed used by shuffling procedure.
    verbose : bool, optional, default=True
        Whether to print progress messages to stderr.
    
    Attributes
    ----------
    data : dict
        The main data structure holding information about features
        and feature sets. Populated by calling `compute_index` method.
        Keys of `data` are names of reference
        feature set collections, and values are dicts with the following
        structure:

        | {
        |     ``"sign"`` : ``{"positive", "negative", "absolute"}``
        |         Taken directly from the input data.
        |     ``"mask"`` : `numpy.array`
        |         Array that says whether each feature pair
                  belongs to at least one shared reference set.
                  Feature pairs are packed into a flat array
                  with ``n_features * (n_features - 1) // 2`` elements (see `numpy.triu_indices`).
                  There are 3 possible values:
        |             ``mask == 1``: features belong to the same reference set.
        |             ``mask == 0``: features do not belong to the same reference set,
                      though they form allowed pair type and both are present in some reference sets.
        |             ``mask == -1``: otherwise.
        |     ``"train_val_mask"`` : `numpy.array`
        |         Array that says whether each feature pair
                  belongs to training or validation set.
                  Same order of pairs as in ``mask``.
                  There are 3 possible values:
        |             ``train_val_mask == 0``: pair belongs to training set.
        |             ``train_val_mask == 1``: pair belongs to validation set.
        |             ``train_val_mask == -1``: all other pairs
        |     ``"features_total_pos"`` : `numpy.ndarray`
        |         2D array of shape ``(3, n_features)``. The first index
                  refers to feature pair type: ``0`` is ``training``,
                  ``1`` is ``validation``, and ``2`` is ``all``.
                  The second index stands for feature index,
                  and the array value says how many reference pairs
                  the feature has. These are ``n_j`` in the paper's terminology.
        |     ``"features_total_neg"`` : `numpy.ndarray`
        |         Same as ``"features_total_pos"`` but for
                  non-reference pairs. These are ``N_j - n_j`` in the paper's terminology.
        |     ``"scoring_features_mask"`` : `numpy.array`
        |         Binary array with ``n_features`` elements that says
                  whether each feature should participate in the computation of
                  enrichment scores. The value is determined based on
                  `min_pairs_to_score` parameter.
        |     ``"high_corr_frac"`` : `float`
        |         Taken directly from the input data.
        |     ``"high_corr_pairs_all"`` : `int`
        |         Absolute number of highly ranked feature pairs (all pairs).
        |     ``"high_corr_pairs_training"`` : `int`
        |         Absolute number of highly ranked feature pairs (training pairs).
        |     ``"high_corr_pairs_validation"`` : `int`
        |         Absolute number of highly ranked feature pairs (validation pairs).
        | }
        
        feature_ids, feature_names, feature_types : np.array
            Arrays of feature IDs, names, and types.
            Populated by `compute_index` method.
        
        metric : {"enrichment-based", "BP@K"}
            Metric for evaluating feature correlations.
            Populated by constructor.
        
        verbose : bool
            Whether to print progress messages.
            Populated by constructor.
    """

    def __init__(
        self,
        df_feature_ann,
        ref_feature_colls,
        shuffle_feature_names=False,
        metric="enrichment-based",
        min_pairs_to_score=1000,
        random_seed=17,
        verbose=True
    ):
        # Do some input checks
        assert (
            len(df_feature_ann) == len(set(df_feature_ann.index))
        ), "Feature annotation data shouldn't contain duplicate feature IDs."

        assert (
            set(df_feature_ann.columns) == {"feature_name", "feature_type"}
        ), "Columns of feature annotation data should be 'feature_name' and 'feature_type'."

        for feature_type in df_feature_ann["feature_type"].unique():
            assert (
                "-" not in feature_type
            ), "Feature types should not contain '-' character."
        
        assert (
            metric in {"enrichment-based", "BP@K"}
        ), f"Metric should be either 'enrichment-based' or 'BP@K', but '{metric}' was found"
        
        assert isinstance(min_pairs_to_score, int), "min_pairs_to_score must be int."
        assert isinstance(random_seed, int), "random_seed must be int."
        assert verbose in {True, False}, "verbose must be True or False."

        self.metric = metric
        self.verbose = verbose

        if shuffle_feature_names is not False:
            # We shuffle feature names within each feature type
            # This way we fully preserve structure of reference sets,
            # but destroy association between feature name and data
            np.random.seed(random_seed)

            if shuffle_feature_names is True:
                feature_types_to_shuffle = df_feature_ann["feature_type"].unique()
            else:
                feature_types_to_shuffle = shuffle_feature_names
            
            for feature_type in feature_types_to_shuffle:
                assert (
                    feature_type in set(df_feature_ann["feature_type"])
                ), f"{feature_type} is present in shuffle_feature_names but not in the feature annotation data."

                type_mask = df_feature_ann["feature_type"] == feature_type
                shuffled_feature_names = df_feature_ann.loc[type_mask, "feature_name"].sample(n=type_mask.sum()).to_list()
                df_feature_ann.loc[type_mask, "feature_name"] = shuffled_feature_names

        self.feature_ids = np.array(df_feature_ann.index.to_list())
        self.feature_names = np.array(df_feature_ann["feature_name"].to_list())
        self.feature_types = np.array(df_feature_ann["feature_type"].to_list())

        # Make dicts to quickly convert between feature *names* and feature indices
        # One name can have many indices, but one index have only one name
        self.feature_name_to_idxs = {}
        self.feature_idx_to_name, self.feature_idx_to_type = [], []
        for feature_idx, feature_id in enumerate(self.feature_ids):
            feature_name = df_feature_ann.at[feature_id, "feature_name"]
            feature_type = df_feature_ann.at[feature_id, "feature_type"]
            self.feature_name_to_idxs[feature_name] = self.feature_name_to_idxs.get(feature_name, []) + [feature_idx]
            self.feature_idx_to_name.append(feature_name)
            self.feature_idx_to_type.append(feature_type)
        self.feature_idx_to_name = np.array(self.feature_idx_to_name)
        self.feature_idx_to_type = np.array(self.feature_idx_to_type)

        # Load reference sets and store them in a nice data structure 
        self.compute_index(ref_feature_colls, min_pairs_to_score)

    def compute_index(self, ref_feature_colls, min_pairs_to_score):
        """
        Populate `data` attribute.
        Input arguments are the same as in the constructor.

        Parameters
        ----------
        ref_feature_colls : dict
            A dictionary with information about reference feature set collections.
        min_pairs_to_score : int
            Minimum number of total pairs containing a feature
            to include the feature in the enrichment score computations.
        """

        n_features = len(self.feature_ids)
        # These arrays contain feature indices for (feature1, feature2) pairs in mask
        self.feature_idxs1, self.feature_idxs2 = np.triu_indices(n_features, k=1)

        self.data = {}

        for ref_feature_coll in ref_feature_colls:
            if self.verbose:
                print(
                    f"{datetime.now()} | Loading {ref_feature_coll} reference collection...",
                    file=sys.stderr
                )

            assert (
                os.path.isfile(ref_feature_colls[ref_feature_coll]["path"])
            ), f"File {ref_feature_colls[ref_feature_coll]['path']} not found."

            db_file = io.BufferedReader(open(ref_feature_colls[ref_feature_coll]["path"], "rb"))
            # Each line has reference set name (field 0) and list of features (fields 2+)
            # n features listed on a line will result in n*(n-1)/2 feature pairs
            
            # Features from all lines are stored in a flat 1D arrays
            # ref_feature_sets_boundaries arrays stores the indices separating
            # features from different lines
            feature_idxs_in_ref_sets = []
            ref_feature_sets_boundaries = [0]

            # These features are present in at least one reference set
            unique_feature_idxs_in_db = set()
            
            for line in db_file:
                splitted = line.decode().strip("\n").split("\t")
                assert (
                    len(splitted) >= 4
                ), f"This line of GMT file contains < 4 fields: '{line.decode()}'."
                feature_names = splitted[2:]
                
                feature_idxs = [
                    feature_idx
                    for feature_name in feature_names if feature_name in self.feature_name_to_idxs
                    for feature_idx in self.feature_name_to_idxs[feature_name]
                ]
                unique_feature_idxs_in_db |= set(feature_idxs)
                if len(feature_idxs) <= 1:
                    continue

                feature_idxs_in_ref_sets += feature_idxs
                ref_feature_sets_boundaries.append(ref_feature_sets_boundaries[-1] + len(feature_idxs))

            db_file.close()

            feature_idxs_in_ref_sets = np.array(feature_idxs_in_ref_sets, dtype="int64")
            ref_feature_sets_boundaries = np.array(ref_feature_sets_boundaries, dtype="int64")
            unique_feature_idxs_in_db = np.array(sorted(list(unique_feature_idxs_in_db)), dtype="int64")
            
            # Numba cannot work with strings and tuples normally
            # So, we encode feature types by numbers
            unique_types, inverse_idx = np.unique(self.feature_idx_to_type, return_inverse=True)
            type_ids = np.arange(len(unique_types))
            feature_idx_to_type_id = type_ids[inverse_idx]

            # We hash type_id tuples using Cantor pairing function
            type_to_id = dict(zip(unique_types, type_ids))
            allowed_feature_pair_types = []
            for pair_type in ref_feature_colls[ref_feature_coll]["feature_pair_types"]:
                assert (
                    "-" in pair_type
                ), f"Allowed pair type must contain '-' character but none is found in '{pair_type}'."
                assert (
                    pair_type.split("-")[0] in type_to_id and
                    pair_type.split("-")[1] in type_to_id
                ), f"One of the allowed feature pair types is not present in feature annotation data: '{pair_type}'."

                k1 = type_to_id[pair_type.split("-")[0]]
                k2 = type_to_id[pair_type.split("-")[1]]
                cantor = (k1 + k2) * (k1 + k2 + 1) // 2 + k2
                allowed_feature_pair_types.append(cantor)
            
            allowed_feature_pair_types = np.array(allowed_feature_pair_types, dtype="int64")
            
            self.data[ref_feature_coll] = {}

            assert (
                ref_feature_colls[ref_feature_coll]["sign"] in {"positive", "negative", "absolute"}
            ), f"Sign can be 'positive', 'negative', or 'absolute', but '{ref_feature_colls[ref_feature_coll]['sign']}' was found."
            self.data[ref_feature_coll]["sign"] = ref_feature_colls[ref_feature_coll]["sign"]

            # Effective data structure for ground truth flag 
            mask = self._compute_ref_feature_coll_mask(
                feature_idxs_in_ref_sets, ref_feature_sets_boundaries, unique_feature_idxs_in_db,
                allowed_feature_pair_types, feature_idx_to_type_id,
                n_features
            )
            assert (
                np.any(mask == 0) and np.any(mask == 1)
            ), (
                f"There are no reference or non-reference feature pairs found in {ref_feature_coll}. "
                "Check that 1) feature names match between feature annotation data "
                "and the GMT file, 2) you correctly specified allowed feature pair types."
            )

            # Get labels of training and validation sets
            train_val_mask = self._compute_train_val_mask(mask)

            # Count total number of reference and non-reference pairs for each feature
            # Counts are training/validation/all-specific
            features_total_pos, features_total_neg, scoring_features_mask = self._compute_feature_totals(
                mask, train_val_mask, self.feature_idxs1, self.feature_idxs2, n_features, min_pairs_to_score
            )
            assert (
                np.any(scoring_features_mask)
            ), (
                f"None of the features can be used for scoring in {ref_feature_coll}. "
                "The value of min_pairs_to_score might be too high for your data."
            )

            self.data[ref_feature_coll]["mask"] = mask
            self.data[ref_feature_coll]["train_val_mask"] = train_val_mask

            self.data[ref_feature_coll]["features_total_pos"] = features_total_pos
            self.data[ref_feature_coll]["features_total_neg"] = features_total_neg
            self.data[ref_feature_coll]["scoring_features_mask"] = scoring_features_mask

            # A fraction of all participating pairs are "highly correlated"
            high_corr_frac = ref_feature_colls[ref_feature_coll]["high_corr_frac"]
            assert (
                isinstance(high_corr_frac, float) and 0 <= high_corr_frac <= 1
            ), "high_corr_frac must be float between 0 and 1."
            total_num_pairs = np.sum(mask != -1)
            high_corr_pairs_all = int(total_num_pairs * high_corr_frac)
            
            n_pairs_all = np.sum(train_val_mask != -1)
            n_pairs_train = np.sum(train_val_mask == 0)
            n_pairs_val = np.sum(train_val_mask == 1)

            self.data[ref_feature_coll]["high_corr_frac"] = high_corr_frac
            self.data[ref_feature_coll]["high_corr_pairs_all"] = high_corr_pairs_all
            self.data[ref_feature_coll]["high_corr_pairs_training"] = int(high_corr_pairs_all * n_pairs_train / n_pairs_all)
            self.data[ref_feature_coll]["high_corr_pairs_validation"] = int(high_corr_pairs_all * n_pairs_val / n_pairs_all)
    
    
    def compute_corr_scores(self, corrs, full_output=False):
        """
        Evaluate feature-feature correlations with respect to the
        reference feature sets.
        
        Parameters
        ----------
        corrs : numpy.array
            Array of feature-feature correlations (flat format).
        full_output : bool, optional, default=False
            Whether to return large arrays with detailed scores
            (not only final scores).

        Returns
        -------
        res: dict
            If `full_output` is ``False``, dict has the following structure:

            | {
            |     ``ref_feature_coll`` : {
            |         ``"score {subset}"`` : `float`
            |     }
            | }

            Here, subset is ``"training"``, ``"validation"``, and ``"all"`` (feature pair subsets).
            Score is defined according to the `metric` attribute.
            If `full_output` is ``True``, the following keys are added
            to the dict corresponding to each ``ref_feature_coll``:

            | {
            |     ``"BPs_at_K {subset}"`` : `numpy.array`
            |         Balanced precisions for each feature.
            |     ``"enrichments {subset}"`` : `numpy.array`
            |         Enrichments for each feature.
            |     ``"pvalues {subset}"`` : `numpy.array`
            |         p-values for each feature.
            |     ``"pvalues_adj {subset}"`` : `numpy.array`
            |         Adjusted p-values for each feature.
            |     ``"TPs_at_K {subset}"`` : `numpy.array`
            |         True positives for each feature (``k_j`` in the notation of paper).
            |     ``"num_pairs {subset}"`` : `numpy.array`
            |         True positives + false positives for each feature (``K_j`` in the notation of paper).
            |     ``"scoring_features_mask"`` : `numpy.array`
            |         Binary array of whether each feature participates in score computation.
            |     ``"corrs"`` : `numpy.array`
            |         Feature-feature correlations.
            |     ``"mask"`` : `numpy.array`
            |         Whether each feature pair belong to at least one shared reference set.
            |     ``"train_val_mask"`` : `numpy.array`
            |         Whether each feature pair belong to training or validation set.
            | }

            The last four arrays are sorted by correlation value
            in a direction specified by the ``"sign"`` value of the `data` attribute.
        """

        res = {}
        # Different reference collections may use the same sorting sign
        # So the sorting will be done only once for each sign and cached
        sort_cache = {}

        for ref_feature_coll in self.data:
            sign = self.data[ref_feature_coll]["sign"]

            if sign in sort_cache:
                corrs_sorted_args = sort_cache[sign]
            else:
                if sign == "negative":
                    corrs_sorted_args = parallel_sort.argsort(corrs)
                elif sign == "positive":
                    corrs_sorted_args = parallel_sort.argsort(-corrs)
                elif sign == "absolute":
                    corrs_sorted_args = parallel_sort.argsort(-np.abs(corrs))
                
                sort_cache[sign] = corrs_sorted_args
            
            (
                BPs_at_K, enrichments, pvalues, TPs_at_K, FPs_at_K,
                corrs_sorted, mask_sorted, train_val_mask_sorted
            ) = self._compute_scores_for_ref_feature_coll(
                corrs, corrs_sorted_args,
                self.data[ref_feature_coll]["mask"],
                self.data[ref_feature_coll]["train_val_mask"],
                self.data[ref_feature_coll]["high_corr_pairs_all"],
                self.data[ref_feature_coll]["high_corr_pairs_training"],
                self.data[ref_feature_coll]["high_corr_pairs_validation"],
                self.data[ref_feature_coll]["features_total_pos"],
                self.data[ref_feature_coll]["features_total_neg"],
                self.data[ref_feature_coll]["scoring_features_mask"],
                self.feature_idxs1,
                self.feature_idxs2,
            )

            # Need to adjust p-values outside of the previous method
            # because numba doesn't support this function
            scoring_mask = self.data[ref_feature_coll]["scoring_features_mask"]
            pvalues_adj = np.zeros(pvalues.shape, dtype="float64")
            for mode in range(3):
                pvalues_adj[mode, scoring_mask] = false_discovery_control(pvalues[mode, scoring_mask])
                pvalues_adj[mode, ~scoring_mask] = np.nan

            neg_log_pvalues_adj = -np.log10(pvalues_adj)
            if not np.all(np.isnan(neg_log_pvalues_adj)):
                avg_log_padj = np.nanmean(neg_log_pvalues_adj, axis=1)
            else:
                avg_log_padj = np.array([0.0, 0.0, 0.0])

            # This is aggregated BP@K score (a single number)
            scoring_features_mask = self.data[ref_feature_coll]["scoring_features_mask"]
            agg_total_pos = np.nansum(
                self.data[ref_feature_coll]["features_total_pos"][:, scoring_features_mask],
                axis=1
            )
            agg_total_neg = np.nansum(
                self.data[ref_feature_coll]["features_total_neg"][:, scoring_features_mask],
                axis=1
            )
            agg_TP_at_K = np.nansum(TPs_at_K[:, scoring_features_mask], axis=1)
            agg_FP_at_K = np.nansum(FPs_at_K[:, scoring_features_mask], axis=1)
            agg_BP_at_K_TP = agg_TP_at_K / (agg_TP_at_K + agg_FP_at_K) * agg_total_neg
            agg_BP_at_K_FP = agg_FP_at_K / (agg_TP_at_K + agg_FP_at_K) * agg_total_pos
            agg_BP_at_K = agg_BP_at_K_TP / (agg_BP_at_K_TP + agg_BP_at_K_FP)

            final_scores = avg_log_padj if self.metric == "enrichment-based" else agg_BP_at_K

            res[ref_feature_coll] = dict(zip(
                ["score training", "score validation", "score all"],
                final_scores
            ))
            
            if full_output:
                for mode, subset in enumerate(["training", "validation", "all"]):
                    res[ref_feature_coll][f"BPs_at_K {subset}"] = BPs_at_K[mode]
                    res[ref_feature_coll][f"enrichments {subset}"] = enrichments[mode]
                    res[ref_feature_coll][f"pvalues {subset}"] = pvalues[mode]
                    res[ref_feature_coll][f"pvalues_adj {subset}"] = pvalues_adj[mode]
                    res[ref_feature_coll][f"TPs_at_K {subset}"] = TPs_at_K[mode]
                    res[ref_feature_coll][f"num_pairs {subset}"] = (TPs_at_K + FPs_at_K)[mode]
                
                res[ref_feature_coll]["scoring_features_mask"] = scoring_features_mask
                res[ref_feature_coll]["corrs"] = corrs_sorted
                res[ref_feature_coll]["mask"] = mask_sorted
                res[ref_feature_coll]["train_val_mask"] = train_val_mask_sorted

        return res
    
    
    @staticmethod
    @njit(parallel=True)
    def _compute_ref_feature_coll_mask(
        feature_idxs_in_ref_sets, ref_feature_sets_boundaries, unique_feature_idxs_in_db,
        allowed_feature_pair_types, feature_idx_to_type_id,
        n_features
    ):
        """
        Efficiently create a flat array that answers
        whether each feature pair belong at least one shared
        reference feature set.

        Parameters
        ----------
        feature_idxs_in_ref_sets : numpy.array
            Array with indices of features that belong to
            each reference sets. Each reference set is
            represented by a consecutive block of feature
            indices.
        ref_feature_sets_boundaries : numpy.array
            Array with indices of `feature_idxs_in_ref_sets`, which
            tell where each referense set starts and ends.
        unique_feature_idxs_in_db : numpy.array
            Array of feature indices that belong to
            at least one reference set.
        allowed_feature_pair_types : numpy.array
            Array of feature pair types that are participating
            in the score computations. Since Numba cannot work
            with string and tuples normally, these should be
            integers computes using Cantor's pairing function,
            see `compute_index` code.
        feature_idx_to_type_id : numpy.array
            Array mapping feature indices to feature types.
            Indices of the array stand for feature indices,
            and values stand for feature type indices.
        n_features : int
            Total number of features.

        Returns
        -------
        mask : numpy.array
            See `data` attribute description.
        """

        # Allocate memory
        n_pairs = n_features * (n_features - 1) // 2
        mask = np.empty((n_pairs,), dtype="int8")
        # Initialize everything with -1
        mask[:] = -1

        # Set 0s for all pairs composed of features from reference sets file
        # the only constraint: feature types
        for p1 in prange(unique_feature_idxs_in_db.shape[0]):
            for p2 in range(p1 + 1, unique_feature_idxs_in_db.shape[0]):
                i = unique_feature_idxs_in_db[p1]
                j = unique_feature_idxs_in_db[p2]

                k1, k2 = feature_idx_to_type_id[i], feature_idx_to_type_id[j]
                cantor1 = (k1 + k2) * (k1 + k2 + 1) // 2 + k1
                cantor2 = (k1 + k2) * (k1 + k2 + 1) // 2 + k2
                if (
                    cantor1 in allowed_feature_pair_types or
                    cantor2 in allowed_feature_pair_types
                ):
                    mask_idx = j - 1 + i * (n_features - 1) - i * (i + 1) // 2
                    mask[mask_idx] = 0
        
        # Then, set 1s
        # We cannot use parallel processing here because of possible race conditions
        for p in range(1, ref_feature_sets_boundaries.shape[0]):
            for idx_pos1 in range(ref_feature_sets_boundaries[p - 1], ref_feature_sets_boundaries[p]):
                for idx_pos2 in range(idx_pos1 + 1, ref_feature_sets_boundaries[p]):
                    i = min(feature_idxs_in_ref_sets[idx_pos1], feature_idxs_in_ref_sets[idx_pos2])
                    j = max(feature_idxs_in_ref_sets[idx_pos1], feature_idxs_in_ref_sets[idx_pos2])

                    k1, k2 = feature_idx_to_type_id[i], feature_idx_to_type_id[j]
                    cantor1 = (k1 + k2) * (k1 + k2 + 1) // 2 + k1
                    cantor2 = (k1 + k2) * (k1 + k2 + 1) // 2 + k2
                    if (
                        cantor1 in allowed_feature_pair_types or
                        cantor2 in allowed_feature_pair_types
                    ):
                        mask_idx = j - 1 + i * (n_features - 1) - i * (i + 1) // 2
                        mask[mask_idx] = 1

        return mask
    
    @staticmethod
    @njit(parallel=True)
    def _compute_train_val_mask(mask):
        """
        Efficiently create a flat array that answers
        whether each feature pair belong to training or
        validation set. The function is deterministic,
        as training and validation set are derived by
        taking each 2nd element.

        Parameters
        ----------
        mask : numpy.array
            Array returned by the `_compute_ref_feature_coll_mask` method.

        Returns
        -------
        train_val_mask : numpy.array
            See `data` attribute description.
        """

        train_val_mask = np.empty(mask.shape, dtype="int8")

        # Assign training and validation sets
        # = 0 stands for train, = 1 stands for validation, = -1 stands for others
        # This code will make sure that
        # a) train/validation separation is deterministic
        # b) it is the same for different reference collections
        train_val_mask[::2] = 1
        train_val_mask[1::2] = 0
        train_val_mask[np.where(mask == -1)[0]] = -1

        return train_val_mask
    
    @staticmethod
    @njit(parallel=True)
    def _compute_feature_totals(
        mask, train_val_mask,
        feature_idxs1, feature_idxs2, n_features,
        min_pairs_to_score
    ):
        """
        Efficiently compute statistics on number
        of feature pairs that involve each individual feature.

        Parameters
        ----------
        mask : numpy.array
            Array returned by the `_compute_ref_feature_coll_mask` method.
        train_val_mask : numpy.array
            Array returned by the `_compute_train_val_mask` method.
        feature_idxs1 : numpy.array
            Indices of the first features in the flat array of pairs.
        feature_idxs2 : numpy.array
            Indices of the second features in the flat array of pairs.
        n_features : int
            Number of features.
        min_pairs_to_score : int
            Same as in the constructor.

        Returns
        -------
        features_total_pos : numpy.ndarray
            See `data` attribute description.
        features_total_neg : numpy.ndarray
            See `data` attribute description.
        scoring_features_mask : numpy.array
            See `data` attribute description.
        """

        # Feature counts are computed separately for
        # trainining (0), validation (1), and all pairs (2)
        features_total_pos = np.zeros((3, n_features), dtype="int64")
        features_total_neg = np.zeros((3, n_features), dtype="int64")
        # These are the features that are involved in at least min_pairs_to_score pairs,
        # including at least one reference and one non-reference pair
        scoring_features_mask = np.ones((n_features,), dtype="bool")

        # train, validaion, all 
        # No prange because of the race condition on scoring_features_mask
        for mode in range(3):
            if mode != 2:
                idxs = np.where(train_val_mask == mode)[0]
            else:
                idxs = np.where(mask != -1)[0]

            # Non-parallel because of race conditions
            # But it is very fast anyway
            for mask_idx in idxs:
                feature1_idx = feature_idxs1[mask_idx]
                feature2_idx = feature_idxs2[mask_idx]
                TP_flag = mask[mask_idx]

                features_total_pos[mode, feature1_idx] += TP_flag
                features_total_pos[mode, feature2_idx] += TP_flag
                features_total_neg[mode, feature1_idx] += 1 - TP_flag
                features_total_neg[mode, feature2_idx] += 1 - TP_flag

            scoring_features_mask = (
                scoring_features_mask & 
                ((features_total_pos[mode] + features_total_neg[mode]) >= min_pairs_to_score)
            )
        
        return features_total_pos, features_total_neg, scoring_features_mask
    
    
    @staticmethod
    @njit(parallel=True)
    def _compute_scores_for_ref_feature_coll(
        corrs, corrs_sorted_args,
        mask, train_val_mask,
        high_corr_pairs_all, high_corr_pairs_train, high_corr_pairs_val,
        features_total_pos, features_total_neg, scoring_features_mask,
        feature_idxs1, feature_idxs2,
        num_regularization_pairs=5,
    ):
        """
        Evaluate feature-feature correlations with respect to
        one reference features set collection.
        
        Parameters
        ----------
        corrs : numpy.array
            Array of feature-feature correlations (flat format).
        corrs_sorted_args : numpy.array
            Array of indices that sort corrs array in needed way.
        mask : numpy.array
            Array returned by the `_compute_ref_feature_coll_mask` method.
        train_val_mask : numpy.array
            Array returned by the `_compute_train_val_mask` method.
        high_corr_pairs_all : int
            See `data` attribute description.
        high_corr_pairs_train : int
            See `data` attribute description.
        high_corr_pairs_val : int
            See `data` attribute description.
        features_total_pos : numpy.ndarray
            Array returned by the `_compute_feature_totals` method.
        features_total_neg : numpy.ndarray
            Array returned by the `_compute_feature_totals` method.
        scoring_features_mask : numpy.array
            Array returned by the `_compute_feature_totals` method.
        feature_idxs1 : numpy.array
            Indices of the first features in the flat array of pairs.
        feature_idxs2 : numpy.array
            Indices of the second features in the flat array of pairs.
        num_regularization_pairs : int, optional, default=5
            Number of feature pairs to add to each ``K_j`` for
            Bayesian regularization purposes.

        Returns
        -------
        BPs_at_K : numpy.array
            Balanced precisions for each feature.
        enrichments : numpy.array
            Enrichments for each feature.
        pvalues : numpy.array
            p-values for each feature.
        TPs_at_K : numpy.array
            True positives for each feature.
        FPs_at_K : numpy.array
            False positives for each feature.
        corrs_filtered : numpy.array
            Sorted feature-feature correlations with ``mask != -1``.
        mask_filtered : numpy.array
            Mask sorted in the same way as `corrs_filtered`.
        train_val_mask_filtered : numpy.array
            Training/validation mask sorted in the same way as `corrs_filtered`.
        """
    
        # Parallel re-ordering of arrays according to order in corrs_sorted_args
        # Numba can't auto parallelize this, so we need to write for loop with prange
        
        # Allocate memory
        corrs_sorted = np.zeros(corrs_sorted_args.shape, dtype="float32")
        mask_sorted = np.zeros(corrs_sorted_args.shape, dtype="int8")
        train_val_mask_sorted = np.zeros(corrs_sorted_args.shape, dtype="int8")
        feature_idxs1_sorted = np.zeros(corrs_sorted_args.shape, dtype="int64")
        feature_idxs2_sorted = np.zeros(corrs_sorted_args.shape, dtype="int64")

        # Re-order
        for i in prange(corrs_sorted_args.shape[0]):
            corrs_sorted[i] = corrs[corrs_sorted_args[i]]
            mask_sorted[i] = mask[corrs_sorted_args[i]]
            train_val_mask_sorted[i] = train_val_mask[corrs_sorted_args[i]]
            feature_idxs1_sorted[i] = feature_idxs1[corrs_sorted_args[i]]
            feature_idxs2_sorted[i] = feature_idxs2[corrs_sorted_args[i]]
        
        # Next, we need to keep only those elements that corresponds to mask != -1
        # Again, we need explicit prange loop to make it parallel

        # Allocate memory
        mask_non_empty_idxs = np.where(mask_sorted != -1)[0]
        corrs_filtered = np.zeros(mask_non_empty_idxs.shape, dtype="float32")
        mask_filtered = np.zeros(mask_non_empty_idxs.shape, dtype="int8")
        train_val_mask_filtered = np.zeros(mask_non_empty_idxs.shape, dtype="int8")
        feature_idxs1_filtered = np.zeros(mask_non_empty_idxs.shape, dtype="int64")
        feature_idxs2_filtered = np.zeros(mask_non_empty_idxs.shape, dtype="int64")
        
        # Select subset of elements
        for i in prange(mask_non_empty_idxs.shape[0]):
            corrs_filtered[i] = corrs_sorted[mask_non_empty_idxs[i]]
            mask_filtered[i] = mask_sorted[mask_non_empty_idxs[i]]
            train_val_mask_filtered[i] = train_val_mask_sorted[mask_non_empty_idxs[i]]
            feature_idxs1_filtered[i] = feature_idxs1_sorted[mask_non_empty_idxs[i]]
            feature_idxs2_filtered[i] = feature_idxs2_sorted[mask_non_empty_idxs[i]]
        
        # Next, we calculate precision for neighborhood of each feature
        n_features1 = np.max(feature_idxs1) + 1
        n_features2 = np.max(feature_idxs2) + 1
        n_features = max(n_features1, n_features2)
        scoring_features_idxs = np.where(scoring_features_mask)[0]
        
        # We calculate precision separately for
        # train (0), validation (1), and all pairs (2)
        TPs_at_K = np.zeros((3, n_features), dtype="int64")
        FPs_at_K = np.zeros((3, n_features), dtype="int64")

        BPs_at_K = np.zeros((3, n_features), dtype="float32")
        enrichments = np.zeros((3, n_features), dtype="float32")
        pvalues = np.zeros((3, n_features), dtype="float64")
        BPs_at_K[:, :] = np.nan
        enrichments[:, :] = np.nan
        pvalues[:, :] = np.nan

        # This is an array to store mean values across features
        #MBPs_at_K = np.zeros((3,), dtype="float32")

        # train, validaion, all 
        for mode in range(3):
            if mode != 2:
                idxs = np.where(train_val_mask_filtered == mode)[0]
                if mode == 0:
                    top_K = high_corr_pairs_train
                else:
                    top_K = high_corr_pairs_val
            else:
                idxs = np.arange(mask_filtered.shape[0])
                top_K = high_corr_pairs_all

            # Non-parallel because of race conditions
            # But it is very fast anyway
            for mask_idx in idxs[:top_K]:
                feature1_idx = feature_idxs1_filtered[mask_idx]
                feature2_idx = feature_idxs2_filtered[mask_idx]
                TP_flag = mask_filtered[mask_idx]

                TPs_at_K[mode, feature1_idx] += TP_flag
                TPs_at_K[mode, feature2_idx] += TP_flag
                FPs_at_K[mode, feature1_idx] += 1 - TP_flag
                FPs_at_K[mode, feature2_idx] += 1 - TP_flag

            ##########################################
            # Balanced precision at K and enrichment #
            ##########################################
        
            # Weights are inverse-proportional to the fraction of positives/negatives
            # Factor of 0.5 is needed to bring TP * TP_weight and FP * FP_weight
            # to the scale needed for regularization
            features_total = features_total_pos[mode, scoring_features_mask] + features_total_neg[mode, scoring_features_mask]
            frac_pos = features_total_pos[mode, scoring_features_mask] / features_total
            frac_neg = features_total_neg[mode, scoring_features_mask] / features_total
            # Bayesian regularization (pseudo-counts)
            regularization_TPs = num_regularization_pairs * frac_pos
            regularization_FPs = num_regularization_pairs * frac_neg

            # Compute BPs_at_K and enrichments
            # j goes over indices of scoring features only
            # feature_idx points to the global index of feature
            for j in prange(scoring_features_idxs.shape[0]):
                feature_idx = scoring_features_idxs[j]
                # For features without any pairs within top-K, this will set
                # BP to 0.5 because of regularization
                # For features with frac_pos or frac_neg == 0, we just set value of 0.5
                if frac_pos[j] != 0 and frac_neg[j] != 0:
                    BPs_at_K[mode, feature_idx] = (
                        ((TPs_at_K[mode, feature_idx] + regularization_TPs[j]) / frac_pos[j]) /
                        (
                            (TPs_at_K[mode, feature_idx] + regularization_TPs[j]) / frac_pos[j] +
                            (FPs_at_K[mode, feature_idx] + regularization_FPs[j]) / frac_neg[j]
                        )
                    )
                else:
                    BPs_at_K[mode, feature_idx] = 0.5

                # Set enrichment to 1 if expected = 0
                if frac_pos[j] != 0:
                    expected = frac_pos[j] * (
                        TPs_at_K[mode, feature_idx] + FPs_at_K[mode, feature_idx] +
                        num_regularization_pairs
                    )
                    enrichments[mode, feature_idx] = (
                        (TPs_at_K[mode, feature_idx] + regularization_TPs[j]) /
                        expected
                    )
                else:
                    enrichments[mode, feature_idx] = 1.0

            ##################################
            # p-values (hypergeometric test) #
            ##################################

            pvalues[mode, scoring_features_mask] = hypergeom_pvalues(
                features_total + num_regularization_pairs,
                features_total_pos[mode, scoring_features_mask] + regularization_TPs.astype("int64"),
                TPs_at_K[mode, scoring_features_mask] + FPs_at_K[mode, scoring_features_mask] + num_regularization_pairs,
                TPs_at_K[mode, scoring_features_mask] + regularization_TPs.astype("int64")
            )

        return (
            BPs_at_K, enrichments, pvalues,
            TPs_at_K, FPs_at_K,
            corrs_filtered, mask_filtered, train_val_mask_filtered
        )