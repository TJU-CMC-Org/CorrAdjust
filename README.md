# CorrAdjust

![test package](https://github.com/TJU-CMC-Org/CorrAdjust/actions/workflows/python-package.yml/badge.svg)

This is official repository for CorrAdjust Python module.

Correcting for confounding variables is often overlooked when computing correlations between data features, even though it can profoundly affect results. We introduce CorrAdjust, a method for identifying and removing such hidden confounders. CorrAdjust selects a subset of principal components to eliminate from the data being processed by maximizing the enrichment of "reference pairs" among highly correlated feature pairs. Unlike traditional machine learning metrics, this novel enrichment-based metric is specifically designed to evaluate correlation data and provides valuable feature-level interpretation.

**Documentation:** [https://tju-cmc-org.github.io/CorrAdjust](https://tju-cmc-org.github.io/CorrAdjust).

Read more in our preprint:
> Nersisyan S, Loher P, Rigoutsos I. CorrAdjust unveils biologically relevant transcriptomic correlations by efficiently eliminating hidden confounders. bioRxiv \[Preprint\]. 2024 Dec 25:2024.12.24.630258. doi: [10.1101/2024.12.24.630258](https://doi.org/10.1101/2024.12.24.630258).
