# PYTHONLIB = python installation directory if use 'make install'
# make sure this is part of PYTHONPATH (in .bashrc, eg)
PIP = python -m pip
PYTHON = python
PYTHONVERSION = python`python -c 'import platform; print(platform.python_version())'`
VERSION = `python -c 'import qcdevol; print(qcdevol.__version__)'`

SRCFILES := $(shell ls setup.py pyproject.toml src/qcdevol/*.py)
DOCFILES := $(shell ls doc/*.rst doc/conf.py)

install-user :
	python make_version.py src/qcdevol/_version.py
	$(PIP) install . --user --no-cache-dir

install install-sys :
	python make_version.py src/qcdevol/_version.py
	$(PIP) install . --no-cache-dir

uninstall :			# mostly works (may leave some empty directories)
	$(PIP) uninstall qcdevol

update:
	make uninstall install


.PHONY : doc

doc/html/index.html : $(SRCFILES) $(DOCFILES) setup.cfg
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
	pip wheel .
	mv qcdevol*whl dist/
	rm *.whl


upload-twine: $(CYTHONFILES)
	twine upload dist/qcdevol-$(VERSION).tar.gz

upload-git: $(CYTHONFILES)
	echo  "version $(VERSION)"
	make doc-html
	git diff --exit-code
	git diff --cached --exit-code
	git push origin master

tag-git:
	echo  "version $(VERSION)"
	git tag -a v$(VERSION) -m "version $(VERSION)"
	git push origin v$(VERSION)

test-download:
	-$(PIP) uninstall qcdevol
	$(PIP) install qcdevol --no-cache-dir
