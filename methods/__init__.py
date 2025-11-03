# methods/__init__.py
# Register plugins via side-effect imports (Python 3.9)

# identity baseline
from .baselines.identity import plugin as _id_plugin  # registers "identity"

# ToMe adapter
from .tome_adapter import plugin as _tome_plugin      # registers "tome"

# Ours
from .ours import *                                   # registers "ours"
