# methods/tome/schedule.py
# r-per-layer schedule (uniform). We interpret r as "tokens merged per selected block".

def r_for_block(r_per_layer, block_idx, selected_layers=None):
    """
    Return r to apply for this block.
    - r_per_layer: int, target merges per selected block
    - selected_layers: list[int] or None (None => all blocks selected)
    """
    if r_per_layer <= 0:
        return 0
    if selected_layers is None:
        return int(r_per_layer)
    return int(r_per_layer) if (block_idx in set(selected_layers)) else 0

def feasible_r(r, num_tokens_no_cls):
    """
    Cap r so that we do not merge more than half of the available non-CLS tokens.
    """
    return max(0, min(int(r), max(0, num_tokens_no_cls // 2)))
