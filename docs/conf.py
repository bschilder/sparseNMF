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
    "myst_nb",  # supersedes myst_parser; renders .ipynb with outputs
]

# Notebooks ship with their outputs already baked in (we run them
# locally before committing). Don't re-execute on RTD — saves build
# time and avoids needing torch/CUDA in the docs environment.
nb_execution_mode = "off"

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
# The canonical RTD look — same theme readthedocs.io uses, same fonts
# (Lato body, Roboto Slab headings, Inconsolata code), same sidebar
# layout. See ``docs.yml`` for the GH Pages build pipeline.
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_title = f"{project} {release}"
html_show_sourcelink = False
# Canonical URL so search engines + the RTD theme's "edit on GitHub"
# button know where the published site lives. Update if the deploy
# target changes.
html_baseurl = "https://bschilder.github.io/sparseNMF/"
html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "prev_next_buttons_location": "bottom",
    # RTD theme's canonical accent — blue, same as the hosted RTD
    # default. Override here to make explicit (the theme picks it
    # up from CSS variables otherwise).
    "style_nav_header_background": "#2980b9",
}
# Wire the "Edit on GitHub" button in the top-right of every page —
# this is RTD's ``html_context`` dance.
html_context = {
    "display_github": True,
    "github_user": "bschilder",
    "github_repo": "sparseNMF",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# ── MyST (Markdown) ─────────────────────────────────────────────────
myst_enable_extensions = ["colon_fence", "deflist", "strikethrough"]
myst_heading_anchors = 3
