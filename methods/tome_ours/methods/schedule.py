# methods/schedule.py
from typing import List, Optional

def parse_layers(spec: str, total_blocks: Optional[int] = None) -> List[int]:
    if spec is None:
        return []
    s = str(spec).lower().strip()
    if s == "" or s == "all":
        return list(range(int(total_blocks))) if total_blocks is not None else []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))

def _protected_count(class_token: bool = True, distill_token: bool = False) -> int:
    n = 0
    if class_token:
        n += 1
    if distill_token:
        n += 1
    return n

def feasible_r(num_tokens: int, r: int, class_token: bool = True, distill_token: bool = False) -> int:
    if num_tokens <= 0 or r <= 0:
        return 0
    max_r = max(0, (int(num_tokens) - _protected_count(class_token, distill_token)) // 2)
    r = int(r)
    if r < 0:
        r = 0
    if r > max_r:
        r = max_r
    return r

def r_for_block(current_tokens: int, r_global: int, class_token: bool = True, distill_token: bool = False) -> int:
    return feasible_r(current_tokens, r_global, class_token, distill_token)
