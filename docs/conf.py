"""Configure the knee-mesher documentation build."""

import sys
from importlib.metadata import version as distribution_version
from pathlib import Path

project = "knee-mesher"
author = "Kalin Gibbons"
copyright = "2020-2026, Kalin Gibbons"
release = distribution_version(project)
version = release

needs_sphinx = "8.2"
extensions = [
    "sphinx.ext.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
]

apidoc_modules = [
    {
        "path": str(Path(__file__).resolve().parents[1] / "src/kneemesher"),
        "destination": "api",
        "implicit_namespaces": True,
        "exclude_patterns": ["**/data/**"],
    },
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".venv"]
todo_emit_warnings = True
pygments_style = "sphinx"

html_theme = "pydata_sphinx_theme"
html_theme_options = {"github_url": "https://github.com/bsu-mbe-cbl/knee-mesher"}
html_static_path = ["_static"]
htmlhelp_basename = "knee-mesher-doc"
latex_documents = [
    ("index", "user_guide.tex", "knee-mesher Documentation", author, "manual"),
]

python_version = ".".join(map(str, sys.version_info[:2]))
intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": (f"https://docs.python.org/{python_version}", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}
