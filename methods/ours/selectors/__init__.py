# methods/ours/selectors/__init__.py
# Import all selector modules so they self-register to the registry.
from . import registry  # ensure registry is loaded
from . import ff        # registers "ff"
from . import topk      # registers "topk"
from . import random    # registers "random"
from . import facility  # registers "facility"
from . import kdpp      # registers "kdpp"
from . import hquota_ff # registers "hquota_ff"
