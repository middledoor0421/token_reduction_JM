# methods/ours/plugin.py
# Ours: fixed-r token reduction per selected block.
# - Order: norm1 -> attention -> build signature (inflow) -> select keep_k -> drop/merge -> residual -> norm2 -> mlp
# - Signature for selection/coverage is 2D per batch: [N-1, H]
# - We DO NOT reshape with batch dimension inside the per-batch loop.

from core.registry import TokenReducerPlugin, register
from .selectors.registry import select, list_selectors
from core.metrics import kcenter_radius, facility_location, pairwise_diversity

import torch
import torch.nn.functional as F

@register("ours")
class OursPlugin(TokenReducerPlugin):
    def __init__(self, cfg):
        super().__init__(cfg)
        # selector & fixed-r config
        self.selector = str(self.cfg.get("selector", "hquota_ff"))
        self.r = int(self.cfg.get("r", 6))  # number of tokens to REMOVE per selected layer (CLS excluded)

        # layers
        layers_str = str(self.cfg.get("layers", "0,1,2"))
        layers_str = layers_str.strip()
        self._sel_layers = [] if layers_str == "" else [int(x) for x in layers_str.split(",") if x.strip() != ""]

        # other configs
        self.cand_extra = int(self.cfg.get("cand_extra", 128))
        self.hq_q = float(self.cfg.get("hq_q", 0.3))
        self.gamma = float(self.cfg.get("gamma", 0.0))
        self.drop_only = bool(self.cfg.get("drop", False))
        self.log_coverage = bool(self.cfg.get("log_coverage", True))

        # runtime states
        self._sizes = None
        self._n_updates = 0
        self._rad_sum = 0.0
        self._fl_sum = 0.0
        self._div_sum = 0.0

        if self.selector not in list_selectors():
            raise ValueError("Unknown selector '%s'" % self.selector)

    def attach(self, model):
        assert hasattr(model, "blocks"), "[ours] model must have 'blocks'"
        self.model = model
        if len(self._sel_layers) == 0:
            self._sel_layers = list(range(len(model.blocks)))  # treat "" as all

        for li, blk in enumerate(model.blocks):
            if li in set(self._sel_layers):
                blk.forward = self._wrap_block_forward(blk, li)

        print(f"[plugin/ours] attached selector={self.selector}, layers={self._sel_layers}, r={self.r}, drop_only={self.drop_only}")

    def finalize(self):
        if self.log_coverage and self._n_updates > 0:
            print(f"[plugin/ours] coverage kcenter={self._rad_sum/self._n_updates:.4f} "
                  f"FL={self._fl_sum/self._n_updates:.2f} div={self._div_sum/self._n_updates:.4f}")
        print("[plugin/ours] done.")

    def _wrap_block_forward(self, blk_ref, block_idx):
        def forward_ours(x):
            # Pre-LN
            x_res = x
            x_norm = blk_ref.norm1(x)  # [B,N,C]

            B, N, C = x_norm.shape
            H = blk_ref.attn.num_heads
            # init size buffer
            if (self._sizes is None) or (self._sizes.size(0) != B) or (self._sizes.size(1) != N):
                self._sizes = torch.ones(B, N, device=x_norm.device, dtype=x_norm.dtype)

            # ===== Attention =====
            qkv = blk_ref.attn.qkv(x_norm)  # [B,N,3C]
            D = (qkv.size(-1) // 3) // H
            q, k, v = qkv.view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)  # [3,B,H,N,D]
            scale = getattr(blk_ref.attn, "scale", D ** -0.5)
            q = q * scale
            logits = torch.matmul(q, k.transpose(-2, -1))  # [B,H,N,N]
            A = F.softmax(logits, dim=-1)
            A = blk_ref.attn.attn_drop(A)
            out = torch.matmul(A, v)  # [B,H,N,D]
            attn_out = out.transpose(1, 2).reshape(B, N, C)
            attn_out = blk_ref.attn.proj(attn_out)
            attn_out = blk_ref.attn.proj_drop(attn_out)

            # ===== Build signature from inflow (2D per batch) =====
            inflow = A.sum(dim=2)              # [B,H,N]  inflow per head
            sig = inflow.transpose(1, 2)       # [B,N,H]  token x head
            # we will pass sig_b: [N-1,H] to selectors/metrics

            # ===== Fixed-r reduction: keep_k = (N-1) - r =====
            n_no_cls = max(0, N - 1)
            keep_k = max(1, n_no_cls - self.r)

            kept_masks = []
            batch_rad = 0.0
            batch_fl = 0.0
            batch_div = 0.0

            for b in range(B):
                # 2D signature for this sample (CLS excluded)
                sig_b = F.normalize(sig[b, 1:, :], dim=-1, eps=1e-6)  # [N-1,H]
                sizes_b = self._sizes[b, 1:]                           # [N-1]

                idx = select(
                    name=self.selector,
                    sig=sig_b,          # 2D
                    keep_k=keep_k,      # number of tokens to KEEP among N-1
                    sizes=sizes_b,
                    cand_extra=self.cand_extra,
                    hq_q=self.hq_q,
                    gamma=self.gamma
                )

                # coverage metrics expect 2D [N',H] and 1D indices
                if self.log_coverage:
                    batch_rad += kcenter_radius(sig_b, idx)
                    batch_fl  += facility_location(sig_b, idx)
                    batch_div += pairwise_diversity(sig_b, idx)

                # build keep mask including CLS=0
                keep_idx = torch.cat(
                    [
                        torch.zeros(1, device=idx.device, dtype=torch.long),  # CLS
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

            # ===== Drop or size-weighted merge (align residual with same mask) =====
            if self.drop_only:
                attn_out = attn_out[kept].view(B, -1, C)
                x_res    = x_res[kept].view(B, -1, C)
                self._sizes = self._sizes * kept.to(dtype=self._sizes.dtype)
            else:
                # assign non-kept -> nearest kept in signature space
                out_mod = attn_out.clone()
                for b in range(B):
                    keep_idx = torch.nonzero(kept[b], as_tuple=False).flatten()
                    non_idx  = torch.nonzero(~kept[b], as_tuple=False).flatten()
                    if non_idx.numel() == 0:
                        continue
                    sig_b = F.normalize(sig[b], dim=-1, eps=1e-6)  # [N,H]
                    S = torch.matmul(sig_b[non_idx], sig_b[keep_idx].t())  # [#non,#keep]
                    assign = S.argmax(dim=-1)

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

            # ===== Residual + MLP =====
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
        return forward_ours
