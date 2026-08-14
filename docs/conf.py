# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# -- Path setup ----------------------------------------------------------------
# Point at the package source so autodoc can import mmirage without installing.
sys.path.insert(0, os.path.abspath("../src"))

# -- Lightweight datasets mock ------------------------------------------------
# The real `datasets` library imports pandas/pyarrow chains that may not be
# present in the docs build environment.  We pre-inject a minimal mock that
# exposes actual Python *classes* (not MagicMock instances) for Dataset and
# DatasetDict so that the PEP-604 union ``Dataset | DatasetDict`` in
# mmirage.core.loader.base works without a TypeError.


class _FakeDataset:
    """Stand-in for datasets.Dataset."""


class _FakeDatasetDict(dict):
    """Stand-in for datasets.DatasetDict."""


_datasets_mock = MagicMock()
_datasets_mock.Dataset = _FakeDataset
_datasets_mock.DatasetDict = _FakeDatasetDict
_datasets_mock.concatenate_datasets = MagicMock(return_value=_FakeDataset())
_datasets_mock.load_from_disk = MagicMock(return_value=_FakeDataset())

sys.modules["datasets"] = _datasets_mock
sys.modules["datasets.arrow_dataset"] = MagicMock()
sys.modules["datasets.dataset_dict"] = MagicMock()

# -- typing.override shim for Python < 3.12 ------------------------------------
# `override` was added to `typing` in Python 3.12.  The source uses it without
# a try/except in some files, so we inject a no-op shim before importing.
import typing as _typing

if not hasattr(_typing, "override"):

    def _override(f):  # type: ignore[return]
        return f

    _typing.override = _override  # type: ignore[attr-defined]

# -- Project information -------------------------------------------------------
project = "MMIRAGE"
release = "0.1.4"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "html_admonition",
    "html_image",
]
myst_heading_anchors = 3

# -- Autodoc configuration -----------------------------------------------------
# Mock heavy runtime dependencies so autodoc can import the package without them.
autodoc_mock_imports = [
    # Heavy ML / inference libs
    "sglang",
    "transformers",
    "torch",
    # Async / server libs
    "pyzmq",
    "uvloop",
    "fastapi",
    "openai",
    "partial_json_parser",
    "sentencepiece",
    "sgl_kernel",
    "compressed_tensors",
    "msgspec",
    "nest_asyncio",
    "xgrammar",
    # Data / serialization (datasets is pre-mocked via sys.modules above)
    "datasets",
    "pyarrow",
    "fsspec",
    "dacite",
    "pydantic",
    "json_repair",
    # Utilities present in pyproject but may be absent locally
    "jmespath",
    "jinja2",
    "PIL",
    "yaml",
    "numpy",
    "huggingface_hub",
    "humanize",
    "tqdm",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_attr_annotations = True

# Suppress noisy-but-benign warnings:
#  - duplicate member descriptions caused by __init__.py re-exports
#  - unresolvable forward refs in mocked type annotations
#  - autodoc.import_object: modules that cannot be imported in doc env
suppress_warnings = [
    "ref.duplicate",
    "sphinx_autodoc_typehints.forward_reference",
    "myst.header",
    "autodoc",
]

# -- HTML output ---------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_title = "MMIRAGE"
html_logo = "_static/logo.svg"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
