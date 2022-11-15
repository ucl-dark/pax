# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Uncomment when apidoc is set
# APICODEDIR = ../pax
# APICMD = sphinx-apidoc
# APIDOCDIR = api
# APIOPTS = ../pax/ ../pax/agents/ ../pax/envs/ 

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	rm -rf "$(BUILDDIR)"
	# @$(APICMD) -e -f -o  "$(APIDOCDIR)" --module-first "$(APICODEDIR)" $(APIOPTS)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
