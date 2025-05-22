import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.moe.layer import MoE
from deepspeed.comm import get_rank
from flash_attn.modules.mha import MHA as FlashMHA

class TransformerLayerMOE(nn.Module):
    """
    A single transformer layer with a MixtureofExperts FFN, using batch_first=True.
    Args:
      dim:                 hidden size
      num_heads:           attention heads
      ffn_dim:             inner FFN dimension
      num_experts:         number of experts
      top_k:               how many experts each token routes to
      dropout:             dropout rate
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 num_experts: int,
                 top_k: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = FlashMHA(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        expert = nn.Sequential(
                nn.Linear(dim, ffn_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, dim),
                nn.Dropout(dropout),
            ).to(get_rank())
        self.moe = MoE(
            hidden_size=dim,
            expert=expert,
            k=top_k,
            num_experts=num_experts,
            ep_size=8,
            use_residual=True
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, dim)
        returns: same shape
        """
        # ---- 1) Self‑Attention block ----
        # normalize, attend, drop, residual
        x_norm = self.norm1(x)  
        attn_out = self.attn(x_norm)  # batch_first=True => returns (batch, seq_len, dim)
        x = x + self.drop1(attn_out)

        # ---- 2) MOE‑FFN block ----
        # normalize, route through MOE, drop, residual
        x_norm = self.norm2(x)
        ff_out, _, _ = self.moe(x_norm)                      # expects (batch, seq_len, dim)
        x = x + self.drop2(ff_out)

        return x
