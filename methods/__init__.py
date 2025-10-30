# methods/__init__.py
# Optional: from .schedule import parse_layers, feasible_r, r_for_block

# Register plugins via side-effects
from .tome_adapter import plugin as _tome_plugin      # registers "tome"
from .baselines.identity import plugin as _id_plugin  # registers "identity"

# If you have our method plugin, keep it:
# from .ours import plugin as _ours_plugin            # registers "ours"
