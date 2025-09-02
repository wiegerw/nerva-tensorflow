# Minimal Sphinx configuration for nerva_tensorflow docs
import os
import sys
from datetime import datetime

# Add src to path so autodoc can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

project = 'nerva_tensorflow'
author = 'Wieger Wesselink and contributors'
current_year = datetime.now().year
copyright = f'{current_year}, {author}'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

# -----------------------------
# Mock heavy or unavailable imports (avoid installing TensorFlow on CI)
# -----------------------------
autodoc_mock_imports = [
    'tensorflow',
    'sklearn',
]

# Move type hints into description to avoid evaluation errors on mocked objects
autodoc_typehints = "description"

# -----------------------------
# Autosummary
# -----------------------------
autosummary_generate = True  # generate _autosummary files automatically

# -----------------------------
# Napoleon settings for Google/NumPy style docstrings
# -----------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -----------------------------
# HTML output
# -----------------------------
html_theme = 'sphinx_rtd_theme'

# -----------------------------
# Autodoc ordering and defaults
# -----------------------------
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# -----------------------------
# Optional: Speed up builds
# -----------------------------
# If desired, skip inherited members from mocked base classes
autodoc_inherit_docstrings = False

