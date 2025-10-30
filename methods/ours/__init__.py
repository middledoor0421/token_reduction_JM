# methods/ours/__init__.py
# Ensure selectors are imported (registration happens) before plugin is loaded.
from .selectors import *   # triggers registration via __init__.py side-effects
from . import plugin       # loads the OursPlugin
