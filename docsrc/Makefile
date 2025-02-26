# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
HTMLDIR       = ../
DOCTREESDIR   = _build/doctrees
FINALDOCDIR   = ../docs


html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(HTMLDIR)" -d "$(DOCTREESDIR)" $(SPHINXOPTS) $(O)
	mv $(HTMLDIR)/html $(FINALDOCDIR)

clean:
	rm -rf $(FINALDOCDIR) $(DOCTREESDIR)
