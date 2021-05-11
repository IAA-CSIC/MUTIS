# Makefile with some convenient quick ways to do common things

PROJECT = mutis

help:
	@echo ''
	@echo '     help               Print this help message (the default)'
	@echo ''
	@echo '     clean              Remove generated files'
	@echo '     clean-repo         Remove all untracked files and directories (use with care!)'
	@echo ''
	@echo '     test               Run pytest'
	@echo '     test-cov           Run pytest with coverage'
	@echo ''
	@echo '     docs-sphinx        Build docs (Sphinx only)'
	@echo '     docs-show          Open local HTML docs in browser'
	@echo ''

clean:
	rm -rf build dist docs/_build docs/api temp/ \
	  htmlcov MANIFEST v mutis.egg-info .eggs .coverage .cache .pytest_cache
	find . -name ".ipynb_checkpoints" -prune -exec rm -rf {} \;
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find mutis -name '*.c' -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

clean-repo:
	@git clean -f -x -d

test:
	python -m pytest -v mutis

test-cov:
	python -m pytest -v mutis --cov=mutis --cov-report=html

docs-sphinx:
	cd docs && python -m sphinx . _build/html -b html

docs-show:
	open docs/_build/html/index.html
