from corradjust import CorrAdjust
import pytest
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image


@pytest.fixture(scope="module")
def corradjust_res():
    """
    Generate data to use in tests.

    Returns
    -------
    dict
        Dict with keys "confounder_PCs", "out_dir", "ref_dir".
    """

    # Load test input data
    script_dir = os.path.dirname(os.path.abspath(__file__))

    df_data_train = pd.read_csv(f"{script_dir}/test_data/df_data_train.tsv", sep="\t", index_col=0)
    df_data_test = pd.read_csv(f"{script_dir}/test_data/df_data_test.tsv", sep="\t", index_col=0)
    df_feature_ann = pd.read_csv(f"{script_dir}/test_data/df_feature_ann.tsv", sep="\t", index_col=0)

    ref_feature_sets = {
        "TestFeatureSets": {
            "path": f"{script_dir}/test_data/ref_feature_sets.gmt",
            "sign": "positive",
            "feature_pair_types": ["mRNA-mRNA"],
            "high_corr_frac": 0.05,
        },
    }

    # Run CorrAdjust
    out_dir = "corradjust_output"
    ref_dir = f"{script_dir}/test_data/corradjust_output"

    print("", file=sys.stderr)
    model = CorrAdjust(
        df_feature_ann,
        ref_feature_sets,
        out_dir,
        winsorize=None,
        min_pairs_to_score=100
    )
    model.fit(df_data_train, n_PCs=10)

    for df_data, samples_subset in [(df_data_train, "training"), (df_data_test, "test")]:
        df_data_clean, _ = model.transform(df_data)
        df_data_clean.to_csv(f"{out_dir}/data_clean.{samples_subset}_samples.tsv", sep="\t")

        model.export_corrs(df_data, f"corrs.{samples_subset}_samples.tsv")

        for pairs_subset in ["training", "validation", "all"]:
            feature_scores = model.compute_feature_scores(df_data, pairs_subset=pairs_subset)
            
            model.make_volcano_plot(feature_scores, f"volcano.{samples_subset}_samples.{pairs_subset}_pairs.png")

            os.makedirs(f"{out_dir}/scores", exist_ok=True)
            for state in ["Raw", "Clean"]:
                df_scores = feature_scores[state]["TestFeatureSets"]
                df_scores.to_csv(
                    f"{out_dir}/scores/{samples_subset}_samples.{pairs_subset}_pairs.{state}.tsv",
                    sep="\t"
                )
            
            model.make_corr_distr_plot(
                df_data,
                f"corr_distr.{samples_subset}_samples.{pairs_subset}_pairs.png",
                pairs_subset=pairs_subset
            )

    return {
        "confounder_PCs": set(model.confounder_PCs),
        "out_dir": out_dir,
        "ref_dir": ref_dir
    }

def assert_images_almost_equal(img_path1, img_path2, label):
    """
    Test that two images are >=90% equal

    Parameters
    ----------
    img_path1, img_path2 : str
        Paths to the images.
    label : str
        Plot name to use in assert statement.
    """

    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)

    # Convert to same mode
    img2 = img2.convert(img1.mode)

    # Convert images to numpy arrays
    img1, img2 = np.asarray(img1), np.asarray(img2)
    assert img1.shape == img2.shape, f"Resolution of {label} differs from reference"

    n_all_pixels = img1.shape[0] * img1.shape[1]
    n_equal_pixels = np.sum(np.min(img1 == img2, axis=2))

    # There might me differences because of font rendering in different maptlotlib versions
    assert n_equal_pixels / n_all_pixels >= 0.9, f"{label} differs from reference"


def test_corradjust_runs(corradjust_res):
    pass


def test_confounder_PCs(corradjust_res):
    assert (
        set(corradjust_res["confounder_PCs"]) == {0, 2, 6}
    ), "Identified confounder PCs are not correct"


def test_fit_tsv(corradjust_res):
    df_fit = pd.read_csv(
        f"{corradjust_res['out_dir']}/fit.tsv",
        sep="\t"
    )
    df_fit_ref = pd.read_csv(
        f"{corradjust_res['ref_dir']}/fit.tsv",
        sep="\t"
    )
    pd.testing.assert_frame_equal(df_fit, df_fit_ref, check_exact=False, atol=1e-5)


def test_data_clean_tsv(corradjust_res):
    for samples_subset in ["training", "test"]:
        df_data_clean = pd.read_csv(
            f"{corradjust_res['out_dir']}/data_clean.{samples_subset}_samples.tsv",
            sep="\t"
        )
        df_data_clean_ref = pd.read_csv(
            f"{corradjust_res['ref_dir']}/data_clean.{samples_subset}_samples.tsv",
            sep="\t"
        )
        pd.testing.assert_frame_equal(df_data_clean, df_data_clean_ref, check_exact=False, atol=1e-5)


def test_corrs_tsv(corradjust_res):
    for samples_subset in ["training", "test"]:
        df_corrs = pd.read_csv(
            f"{corradjust_res['out_dir']}/corrs.{samples_subset}_samples.tsv",
            sep="\t"
        ).sort_values(["feature_id1", "feature_id2"], ignore_index=True)
        df_corrs_ref = pd.read_csv(
            f"{corradjust_res['ref_dir']}/corrs.{samples_subset}_samples.tsv.gz",
            sep="\t"
        ).sort_values(["feature_id1", "feature_id2"], ignore_index=True)
        pd.testing.assert_frame_equal(df_corrs, df_corrs_ref, check_exact=False, atol=1e-5)


def test_scores_tsv(corradjust_res):
    for samples_subset in ["training", "test"]:
        for pairs_subset in ["training", "validation", "all"]:
            for state in ["Raw", "Clean"]:
                df_scores = pd.read_csv(
                    f"{corradjust_res['out_dir']}/scores/{samples_subset}_samples.{pairs_subset}_pairs.{state}.tsv",
                    sep="\t"
                )
                df_scores_ref = pd.read_csv(
                    f"{corradjust_res['ref_dir']}/scores/{samples_subset}_samples.{pairs_subset}_pairs.{state}.tsv",
                    sep="\t"
                )
                pd.testing.assert_frame_equal(df_scores, df_scores_ref, check_exact=False, atol=1e-5)


def test_fit_png(corradjust_res):
    assert_images_almost_equal(
        f"{corradjust_res['out_dir']}/fit.all_samp.png",
        f"{corradjust_res['ref_dir']}/fit.all_samp.png",
        "fit.all_samp.png"
    )


def test_volcano_png(corradjust_res):
    for samples_subset in ["training", "test"]:
        for pairs_subset in ["training", "validation", "all"]:
            assert_images_almost_equal(
                f"{corradjust_res['out_dir']}/volcano.{samples_subset}_samples.{pairs_subset}_pairs.png",
                f"{corradjust_res['ref_dir']}/volcano.{samples_subset}_samples.{pairs_subset}_pairs.png",
                f"volcano.{samples_subset}_samples.{pairs_subset}_pairs.png"
            )


def test_corr_distr_png(corradjust_res):
    for samples_subset in ["training", "test"]:
        for pairs_subset in ["training", "validation", "all"]:
            assert_images_almost_equal(
                f"{corradjust_res['out_dir']}/corr_distr.{samples_subset}_samples.{pairs_subset}_pairs.png",
                f"{corradjust_res['ref_dir']}/corr_distr.{samples_subset}_samples.{pairs_subset}_pairs.png",
                f"corr_distr.{samples_subset}_samples.{pairs_subset}_pairs.png"
            )
