# PYTHONLIB = python installation directory if use 'make install'
# make sure this is part of PYTHONPATH (in .bashrc, eg)
PIP = python -m pip
PYTHON = python
PYTHONVERSION = python`python -c 'import platform; print(platform.python_version())'`
VERSION = `python -c 'import qcdevol; print qcdevol.__version__'`

SRCFILES := $(shell ls setup.py src/qcdevol/*.py)
DOCFILES := $(shell ls doc/*.rst doc/conf.py)

install-user :
	$(PIP) install . --user

install install-sys :
	$(PIP) install .

# $(PYTHON) setup.py install --record files-qcdevol.$(PYTHONVERSION)

uninstall :			# mostly works (may leave some empty directories)
	$(PIP) uninstall qcdevol

update:
	make uninstall install


.PHONY : doc

doc/html/index.html : $(SRCFILES) $(DOCFILES)
	sphinx-build -b html doc/ doc/html

doc-html doc:
	make doc/html/index.html

clear-doc:
	rm  -rf doc/html

.PHONY : tests

tests:
	python -m unittest discover

coverage:
	pytest --cov-report term-missing --cov=qcdevol tests/

sdist:          # source distribution
	$(PYTHON) setup.py sdist


