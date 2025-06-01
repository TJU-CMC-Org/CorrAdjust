:html_theme.sidebar_secondary.remove:

.. corradjust documentation master file, created by
   sphinx-quickstart on Fri Nov  8 15:12:16 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CorrAdjust
##########

This is official documentation for CorrAdjust Python module.

Correcting for confounding variables is often overlooked
when computing correlations between data features,
even though it can profoundly affect results.
We introduce CorrAdjust, a method for identifying and
removing such hidden confounders.
CorrAdjust selects a subset of principal components to eliminate
from the data being processed by maximizing the enrichment of
"reference pairs" among highly correlated feature pairs.
Unlike traditional machine learning metrics,
this novel enrichment-based metric is specifically designed to
evaluate correlation data and provides valuable feature-level interpretation.

Before using CorrAdjust, we strongly recommend reading our paper
to familiarize yourself with the method's key concepts:

   Nersisyan S, Loher P, Rigoutsos I.
   CorrAdjust unveils biologically relevant transcriptomic correlations
   by efficiently eliminating hidden confounders.
   *Nucleic Acids Res.* 2025 May 31;53(10):gkaf444.
   doi: `10.1093/nar/gkaf444 <https://doi.org/10.1093/nar/gkaf444>`_.

Contents
========

.. toctree::
   :maxdepth: 1

   Installation <installation>

.. toctree::
   :maxdepth: 2

   User Guide <tutorial/tutorial>

.. toctree::
   :maxdepth: 1

   API reference <modules/corradjust>

   Source Repository <https://github.com/TJU-CMC-Org/CorrAdjust>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
