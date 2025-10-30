# methods/tome/plugin.py
# ToMe corrected plugin:
# - r per-layer (apply on selected blocks)
# - Matching = Alternating + Greedy 1:1 (see matcher.py), with per-block offset
# - Size-weighted convex merge, CLS preserved
# - ATTENTION → MERGE → MLP order (paper)
# - Proportional attention: add zero-mean log(size) bias of the PREVIOUS block to current logits
# - R_eff logging per block

from core.registry import TokenReducerPlugin, register
from .schedule import r_for_block, feasible_r
from .matcher import match_greedy_bipartite
from .merge import merge_pairs_size_weighted

import torch
import torch.nn.functional as F

@register("tome")
class TomePlugin(TokenReducerPlugin):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.r = int(self.cfg.get("r", 0))                      # r per selected layer
        self.layers = self._parse_layers(self.cfg.get("layers", ""))
        self.match_feature = str(self.cfg.get("match_feature", "k"))  # "k" or "xnorm"
        self.prop_attn = bool(self.cfg.get("prop_attn", False))  # use log(size) bias on next block logits

        # persistent size buffers
        self._sizes = None        # sizes after current merge (used next block)
        self._sizes_prev = None   # sizes snapshot to bias current attention

        # R_eff logging
        self._reff_sum = {}   # block_idx -> total R_eff
        self._reff_cnt = {}   # block_idx -> #updates

    def _parse_layers(self, s):
        if not s:
            return None  # None => all blocks
        return [int(x) for x in str(s).split(",") if x.strip() != ""]

    def attach(self, model):
        assert hasattr(model, "blocks"), "[tome] model must have 'blocks'"
        self.model = model
        for li, blk in enumerate(model.blocks):
            if (self.layers is not None) and (li not in set(self.layers)):
                continue
            blk.forward = self._wrap_block_forward(blk, li)
        print(f"[plugin/tome] attached r-per-layer={self.r}, layers={self.layers or 'all'}, "
              f"match_feature={self.match_feature}, prop_attn={self.prop_attn}")

    def _wrap_block_forward(self, blk_ref, block_idx):
        def forward_tome(x):
            # Pre-LN
            x_res = x
            x_norm = blk_ref.norm1(x)

            B, N, C = x_norm.shape
            # init/resize sizes (current) and sizes_prev (for bias)
            if (self._sizes is None) or (self._sizes.size(0) != B) or (self._sizes.size(1) != N):
                self._sizes = torch.ones(B, N, device=x_norm.device, dtype=x_norm.dtype)
            if (self._sizes_prev is None) or (self._sizes_prev.size(0) != B) or (self._sizes_prev.size(1) != N):
                # first block: no previous merge → sizes_prev = ones (no bias)
                self._sizes_prev = torch.ones(B, N, device=x_norm.device, dtype=x_norm.dtype)

            # ===== Attention (bias with sizes from previous block only) =====
            qkv = blk_ref.attn.qkv(x_norm)                               # [B,N,3C]
            H = blk_ref.attn.num_heads
            D = (qkv.size(-1) // 3) // H
            q, k, v = qkv.view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)     # [3,B,H,N,D]
            scale = getattr(blk_ref.attn, "scale", D ** -0.5)
            q = q * scale
            logits = torch.matmul(q, k.transpose(-2, -1))                 # [B,H,N,N]

            if self.prop_attn:
                # Use sizes from previous block (snapshot before this block's merge)
                bias = torch.log(self._sizes_prev + 1e-6).view(B, 1, 1, N).to(dtype=logits.dtype)
                logits = logits + bias

            A = F.softmax(logits, dim=-1)
            A = blk_ref.attn.attn_drop(A)
            out = torch.matmul(A, v)                                      # [B,H,N,D]
            attn_out = out.transpose(1, 2).reshape(B, N, C)
            attn_out = blk_ref.attn.proj(attn_out)
            attn_out = blk_ref.attn.proj_drop(attn_out)

            # ===== Token selection / merge using fixed r =====
            keep_k = self.r  # fixed number of tokens to keep per layer

            kept_masks = []
            batch_rad = 0.0
            batch_fl = 0.0
            batch_div = 0.0

            # ===== Token selection per batch (exclude CLS) =====
            for b in range(B):
                sig_b = A[b, 1:]  # [N-1,H]
                idx = select(
                    name=self.selector,
                    sig=sig_b,
                    keep_k=keep_k,
                    sizes=self._sizes[b, 1:],  # sizes after previous merge
                    cand_extra=self.cand_extra,
                    hq_q=self.hq_q,
                    gamma=self.gamma
                )

                if self.log_coverage:
                    batch_rad += kcenter_radius(sig_b, idx)
                    batch_fl  += facility_location(sig_b, idx)
                    batch_div += pairwise_diversity(sig_b, idx)

                keep_idx = torch.cat(
                    [
                        torch.zeros(1, device=idx.device, dtype=torch.long),
                        (idx + 1).to(dtype=torch.long, device=idx.device)
                    ],
                    dim=0
                )
                mask = torch.zeros(N, dtype=torch.bool, device=idx.device)
                mask[keep_idx.long()] = True
                kept_masks.append(mask)

            if self.log_coverage and B > 0:
                self._n_updates += 1
                self._rad_sum += float(batch_rad / float(B))
                self._fl_sum  += float(batch_fl  / float(B))
                self._div_sum += float(batch_div / float(B))

            kept = torch.stack(kept_masks, dim=0)  # [B,N] bool

            # ===== Apply drop OR size-weighted merge on attn_out =====
            if self.drop_only:
                attn_out = attn_out[kept].view(B, -1, C)
                x_res    = x_res[kept].view(B, -1, C)
                self._sizes = self._sizes * kept.to(dtype=self._sizes.dtype)
            else:
                v_sig = A  # [B,N,H]
                out_mod = attn_out.clone()
                for b in range(B):
                    keep_idx = torch.nonzero(kept[b], as_tuple=False).flatten()
                    non_idx  = torch.nonzero(~kept[b], as_tuple=False).flatten()
                    if non_idx.numel() == 0:
                        continue
                    S_sel = torch.matmul(v_sig[b][non_idx], v_sig[b][keep_idx].t())  # [#non,#keep]
                    assign = S_sel.argmax(dim=-1)                                    # [#non]
                    s_vec = self._sizes[b]
                    for n_i, k_j in zip(non_idx.tolist(), assign.tolist()):
                        s_k = s_vec[keep_idx[k_j]]
                        s_n = s_vec[n_i]
                        denom = (s_k + s_n + 1e-6)
                        w_k = s_k / denom
                        w_n = s_n / denom
                        out_mod[b, keep_idx[k_j], :] = w_k * out_mod[b, keep_idx[k_j], :] + w_n * out_mod[b, n_i, :]
                        s_vec[keep_idx[k_j]] = s_k + s_n
                    s_vec[non_idx] = 0.0
                attn_out = out_mod[kept].view(B, -1, C)
                x_res    = x_res[kept].view(B, -1, C)

            # ===== Residual add and MLP =====
            if hasattr(blk_ref, "drop_path1") and blk_ref.drop_path1 is not None:
                x_attn = x_res + blk_ref.drop_path1(attn_out)
            elif hasattr(blk_ref, "drop_path") and blk_ref.drop_path is not None:
                x_attn = x_res + blk_ref.drop_path(attn_out)
            else:
                x_attn = x_res + attn_out

            mlp_in = blk_ref.norm2(x_attn)
            mlp_out = blk_ref.mlp(mlp_in)
            if hasattr(blk_ref, "drop_path2") and blk_ref.drop_path2 is not None:
                x = x_attn + blk_ref.drop_path2(mlp_out)
            elif hasattr(blk_ref, "drop_path") and blk_ref.drop_path is not None:
                x = x_attn + blk_ref.drop_path(mlp_out)
            else:
                x = x_attn + mlp_out

            return x
        return forward_tome

    def finalize(self):
        # print per-block average R_eff
        if len(self._reff_cnt) > 0:
            msg = []
            for li in sorted(self._reff_cnt.keys()):
                avg = (self._reff_sum[li] / max(1, self._reff_cnt[li]))
                msg.append(f"L{li}:{avg:.1f}")
            print("[tome] avg R_eff per block → " + " ".join(msg))
        print("[plugin/tome] done.")
