import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import lem.model_utils as model_utils
import lem.test_data as test_data
from lem.trainer import Trainer
from lem.moe import TransformerLayerMOE
from deepspeed.moe.layer import MoE
from deepspeed.moe.utils import is_moe_param
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.ops.adam import DeepSpeedCPUAdam

import torch
import torch.nn as nn

class ConvEmbedder(nn.Module):
    """
    Embed each EEG bins into an embedding vector
    using a 3‑layer dilated 1D‑Conv backbone.
    """
    def __init__(self,
                 in_ch: int = 1,
                 embed_dim: int = 1024,
                 hidden_dims: list = [16, 32, 64],
                 dropout: float = 0.1):
        super().__init__()
        self.conv_backbone = nn.Sequential(
            nn.Conv1d(in_ch,      hidden_dims[0], kernel_size=3,
                      padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(hidden_dims[0]), nn.GELU(), nn.Dropout(dropout),

            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=3,
                      padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(hidden_dims[1]), nn.GELU(), nn.Dropout(dropout),

            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3,
                      padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(hidden_dims[2]), nn.GELU(), nn.Dropout(dropout),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.proj        = nn.Linear(hidden_dims[2], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) → returns (B, 1024)
        """
        h = self.conv_backbone(x)            # (B, hidden_dim, L)
        h = self.global_pool(h).squeeze(-1)  # (B, hidden_dim)
        return self.proj(h)                  # (B, embed_dim)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_bins: int,
        bin_size: int = 10,
        conv_dims: list = [16, 32, 64],
        emb_dim: int = 1024,
        nhead: int = 8,
        num_layers: int = 14,
        dim_feedforward: int = 1024,
        num_experts: int = 30,
        top_k: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bin_size = bin_size
        self.num_bins = num_bins
        self.emb_dim = emb_dim

        # Convolutional embedding for each bin of 1D data
        self.conv = ConvEmbedder(
            in_ch=1,
            embed_dim=emb_dim,
            hidden_dims=conv_dims,
            dropout=dropout
        )

        # Learnable tokens and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_encoding = model_utils.PositionalEncoding(emb_dim, num_bins + 1)

        self.transformer = nn.Sequential(
            *[TransformerLayerMOE(
                dim=emb_dim,
                num_heads=nhead,
                ffn_dim = dim_feedforward,
                num_experts = num_experts,
                top_k = top_k,
                dropout = dropout
            ) for _ in range(num_layers)]
        )
        
        # Initialize tokens & embeddings
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_bins * bin_size)
        mask: (batch, num_bins)
        returns: (batch, num_bins * bin_size)
        """
        batch = x.size(0)
        
        x = x.view(batch, self.num_bins, self.bin_size)
        x_flat = x.view(batch * self.num_bins, 1, self.bin_size)
        emb = self.conv(x_flat).squeeze(-1).view(batch, self.num_bins, self.emb_dim)

        # Find all tokens where the original x is all -100:
        mask = torch.all(x == -100, dim=-1).unsqueeze(-1)
        emb = torch.where(mask, self.mask_token, emb)

        cls = self.cls_token.expand(batch, -1, -1)
        emb = torch.cat([cls, emb], dim=1)
        emb = self.pos_encoding(emb)

        encoded = self.transformer(emb)
        cls_out = encoded[:, 0, :]
        return cls_out

class MLPDecoder(nn.Module):
    def __init__(
        self,
        out_size: int,
        emb_dim: int = 1024,
        decoder_hidden: list = [1024, 1024, 1024],
    ):
        super().__init__()
        self.out_size = out_size
        self.emb_dim = emb_dim
        
        mlp = []
        last_dim = emb_dim
        for hidden in decoder_hidden:
            mlp.append(nn.Linear(last_dim, hidden))
            mlp.append(nn.ReLU(inplace=True))
            last_dim = hidden
        mlp.append(nn.Linear(last_dim, out_size))
        self.decoder = nn.Sequential(*mlp)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, emb_dim)
        returns: (batch, out_size)
        """
        out = self.decoder(x)
        return out

def train_transformer(
    train_loader: DataLoader,
    val_loader: DataLoader,

    bin_size: int = 10,

    # Training parameters
    lr: float = 5e-5,
    lr_ramp: int = 50,
    lr_cycle: int = 100,
    lr_min: float = 1e-6,
    weight_decay: float = 1e-5,

    num_epochs: int = 100,
    device: str = 'cpu',
    
    # Model parameters
    conv_dims: list = [16, 32, 64],
    emb_dim: int = 1024,
    nhead: int = 8,
    num_layers: int = 14,
    dim_feedforward: int = 1024,
    num_experts: int = 30,
    top_k: int = 4,
    dropout: float = 0.1,
    decoder_hidden: list = [1024, 1024, 1024],
    mask_prob: float = 0.1,
    mask_test_prob: float = 0.8,

    args=None,
):
    num_bins = next(iter(train_loader))[0].size(1) // bin_size
    
    # Build model
    enc_model = TransformerEncoder(
        num_bins=num_bins,
        bin_size=bin_size,
        conv_dims=conv_dims,
        emb_dim=emb_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_experts=num_experts,
        top_k=top_k,
        dropout=dropout
    )
    dec_model = MLPDecoder(
        out_size=num_bins * bin_size,
        emb_dim=emb_dim,
        decoder_hidden=decoder_hidden
    )
    model = nn.Sequential(enc_model, dec_model)
    model.to(device)

    parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}

    param_groups = split_params_into_different_moe_groups_for_optimizer(parameters)    
    
    optimizer = DeepSpeedCPUAdam(param_groups, lr=lr, weight_decay=weight_decay)

    lr_scheduler = model_utils.get_cosine_schedule_with_warmup_restarts(
        optimizer,lr_ramp,lr_cycle,1,lr_min)
    

    # Count parameters
    num_params_enc = model_utils.count_parameters(enc_model)
    num_params_dec = model_utils.count_parameters(dec_model)
    num_params = num_params_enc + num_params_dec
    print(f"Number of parameters in Encoder: {num_params_enc}")
    print(f"Number of parameters in Decoder: {num_params_dec}")
    print(f"Total number of parameters: {num_params}")
    
    loss_fn = nn.SmoothL1Loss()
        
    # Trainer
    datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = f"{datetime_str}_transformer"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        scheduler=lr_scheduler,
        log_dir='runs',
        label=label,
        mask_prob=mask_prob,
        mask_len=num_bins,
        mask_bin=bin_size,
        mask_test_prob=mask_test_prob,
        args=args
    )
    best_stats = trainer.fit(
        num_epochs=num_epochs,
        #checkpoint_dir=os.path.join('checkpoints', f'checkpoint_{label}'),
        #checkpoint_interval=10
    )
    return best_stats
