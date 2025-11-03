# methods/ours/selectors/__init__.py
# -*- coding: utf-8 -*-
from typing import Callable, Dict

# 각 selector 구현을 임포트
from .hquota_ff import select_hquota_ff
# 필요하면 다른 selector도 여기서 함께 import 하세요.
# from .topk import select_topk

# 이름 -> 함수 매핑 테이블
_SELECTORS: Dict[str, Callable] = {
    # head-diversity + CLS 보호
    "hquota":    select_hquota_ff,
    "hquota_ff": select_hquota_ff,  # 동의어로 접근 가능
    # "topk":    select_topk,
}

def get_selector(name: str) -> Callable:
    """
    등록된 selector 함수를 이름으로 찾아 반환.
    """
    try:
        return _SELECTORS[name]
    except KeyError:
        # 사용 가능한 키 리스트도 함께 출력
        raise ValueError(f"Unknown selector: {name}. Available: {list(_SELECTORS.keys())}")
