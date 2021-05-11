import sphinx_rtd_theme

extensions = [
    "nbsphinx",
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_templates", "_static", "**.ipynb_checkpoints"]
copyright = "2021 MUTIS Developers"
