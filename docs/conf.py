"""Sphinx configuration for the sparseNMF docs site.

Built locally with ``sphinx-build docs docs/_build/html``; built
automatically by Read the Docs from ``.readthedocs.yaml``. The theme
is the canonical ``sphinx_rtd_theme`` (the default RTD look).
"""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

# ── Project info ────────────────────────────────────────────────────
project = "sparseNMF"
author = "Brian Schilder"
copyright = "2026, Brian Schilder"

try:
    release = _pkg_version("sparse-nmf")
except Exception:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# ── General config ──────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# ── HTML output ─────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_title = f"{project} {release}"
html_show_sourcelink = False
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
}

# ── MyST (Markdown) ─────────────────────────────────────────────────
myst_enable_extensions = ["colon_fence", "deflist", "linkify", "strikethrough"]
myst_heading_anchors = 3
