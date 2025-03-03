FAQ
###


How many samples do I need?
===========================

    We recommend running CorrAdjust with at least 50 samples in the training data (see our paper).


What is the difference between validation and test sets?
========================================================

    Validation set is a set of 50% **feature pairs**; split is automatically performed by CorrAdjust.
    We use these feature pairs to determine the stopping point for the PC optimization,
    while the other 50% of pairs (training pairs) are used to select best PC on each optimization step.
    This is a mandatory procedure which you can't turn off.

    At the same time, test set is user-specified data which have different **samples**
    from the training data. Test set is absolutely optional, and in many cases you don't
    have enough samples to split your data into reasonably-sized (50+ samples)
    training and test sets.


Do I need to use test set?
==========================

    Computing score on a test set is one approach to prove that the PC selection did not overfit.
    At the same time, in our benchmark experiments, we never observed overfitting
    with CorrAdjust (see our paper). If you have enough samples, we recommend using test set to show the absence
    of overfitting, after which you can re-fit a model on the whole data.

    Another way to assess overfitting without test set is to use ``shuffle_feature_names``
    argument of the :class:`CorrAdjust contstructor <corradjust.corradjust.CorrAdjust>`
    option to conduct a negative control experiment.
    With this approach, CorrAdjust will be run on the data with shuffled feature names,
    but exactly the same structure of data table and reference feature sets.


How can I interpret confounder PCs?
===================================

    The confounders identified by CorrAdjust are represented by the PCs and
    do not inherently carry an interpretation.
    However, some of these PCs may be associated with known variables, enhancing interpretability.
    We encourage users to examine these associations
    (e.g., as shown in Figures 2F and 3F of our paper) to better understand what is being corrected.
    See :doc:`standard_run` for a code on how to generate a matrix of confounder PCs.

My samples originate from distinct groups. Should I correct them separately?
============================================================================

    Correcting different sample groups separately could create spurious 
    differential correlations between groups.
    We recommend using the approach shown in :doc:`advanced_run`.
    Specifically, PCs are computed and removed across all samples,
    but enrichment scores are computed separately for each group.
    This option, however, requires a sufficient number of samples
    in each group (we recommend at least 50 samples).


Is CorrAdjust output compatible with other bioinformatics tools?
================================================================

    One of the CorrAdjust output files is a data table after
    confounder removal. This table can be used as input for
    any bioinformatics tool for network analysis (e.g., WGCNA).
    At the same time, it should not be used as input for
    differential expression analysis tools (e.g., DESeq2 or edgeR),
    as CorrAdjust can eliminate true differences in *average* feature
    values between conditions of interest.


Should I use gene- or transcript-level RNA-seq?
===============================================

    Most pathway databases we are aware of operate on gene level.
    Thus, we recommend using gene-level expression data.


Can I use CorrAdjust with non-transcriptomic data?
==================================================

    Current version of CurrAdjust is tested only in mRNA-mRNA
    and miRNA-mRNA settings using RNA-seq data (see our paper).
    At the same time, nothing prevents using it with arbitrary
    numeric dataset if you are interested in reference-guided
    correction of correlations between features.


What if some of my features are not present in the GMT file?
============================================================

    Pairs consisting of such features are labeled with ``flag = -1``
    and are excluded from the computations of enrichments/scores.
    Similarly, features that are present in the GMT file but not in the
    data table are not impacting the results.

