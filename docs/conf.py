import sphinx_rtd_theme

extensions = [
    "nbsphinx",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.viewcode",
]

html_theme = "sphinx_rtd_theme"

automodsumm_inherited_members = True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates", "_static", "**.ipynb_checkpoints"]
copyright = "2021 MUTIS Developers"
project = "MUTIS"
