# methods/__init__.py
# (공용 유틸 export가 필요하면 추가하세요)
# from .schedule import parse_layers, feasible_r, r_for_block

# 플러그인 등록을 위한 사이드이펙트 import
from .tome_adapter import plugin as _tome_adapter_plugin   # registers "tome"
from .baselines.identity import plugin as _identity_plugin # registers "identity"
# 우리 방법론 플러그인도 이미 있다면:
from .ours import plugin as _ours_plugin                   # registers "ours"