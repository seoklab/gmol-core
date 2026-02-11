# ruff: noqa: A001,INP001

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import re

from pydantic import BaseModel
from sphinx.ext import autodoc

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "gmol-core"
copyright = "2026 SNU Compbio Lab"
author = "SNU Compbio Lab"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "ignore-module-all": True,
}
autodoc_inherit_docstrings = False

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "rdkit": ("https://www.rdkit.org/docs", None),
    "nurikit": ("https://nurikit.readthedocs.io/latest", None),
}

add_module_names = False
python_display_short_literal_types = True
python_use_unqualified_type_names = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/seoklab/gmol-core",
            "icon": "fa-brands fa-github",
        }
    ],
    "secondary_sidebar_items": ["page-toc"],
    "footer_start": ["last-updated"],
    "footer_center": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

html_title = project
html_static_path = ["_static"]
html_last_updated_fmt = "%Y-%m-%d %H:%M"

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"


class StrippingDocumenter(autodoc.ClassDocumenter):
    _strip_re = re.compile(r"\s*Bases:\s*:py:class:`(object)`\s*")

    def add_line(self, line: str, source: str, *lineno: int):
        if self._strip_re.fullmatch(line):
            return
        super().add_line(line, source, *lineno)

    def get_object_members(self, want_all: bool):
        flag, members = super().get_object_members(want_all)

        if isinstance(self.object, type) and issubclass(
            self.object, BaseModel
        ):
            members = [
                member
                for member in members
                if not hasattr(BaseModel, member.__name__)
            ]

        return flag, members


autodoc.ClassDocumenter = StrippingDocumenter  # type: ignore[misc]
