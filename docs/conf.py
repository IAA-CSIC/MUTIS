extensions = [
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "numpydoc",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.viewcode",
]


numpydoc_show_class_members = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


# Theme style
# html_style = ''
def setup(app):
    app.add_css_file("mutis.css")


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "_templates", "_static", "**.ipynb_checkpoints"]
copyright = "2021 MUTIS Developers"
project = "MUTIS"
