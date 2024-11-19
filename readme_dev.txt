## Poetry commands

Install and lock dependencies:
conda create -n corradjust_env python=3.10
conda activate corradjust_env
conda install conda-forge::poetry
poetry install

Install dependencies for generating docs
poetry install --extras docs

Build and re-install module locally
poetry build && pip install --no-deps --force-reinstall dist/corradjust-*.whl

## Sphinx commands

You need pandoc installed to convert jupyter tutorial. E.g.,
conda install -c conda-forge pandoc

Auto-generate source files for docs
sphinx-apidoc -e -o modules ../corradjust/
rm modules/modules.rst

Re-build HTML documentation
make clean html

Move docsrc/_build/html/ to docs/ to push docs to GitHub pages.
