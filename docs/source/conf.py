import os
import sys
sys.path.insert(0, os.path.abspath("../../"))

project = "PyCCEA"
copyright = "2025, pedbrgs"
author = "Pedro Vinícius Almeida Borges de Venâncio"

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

autosummary_generate = True
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
numfig = True


templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"

numpydoc_show_class_members = True

html_theme = 'sphinx_rtd_theme'

html_logo = '../figures/logo.png'

html_static_path = ['_static']

html_css_files = [
    'custom.css',
]

html_favicon = '../figures/favicon.png'

html_theme_options = {
    'logo_only': True,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': True,
    'sticky_navigation': False,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
