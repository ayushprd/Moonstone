"""Modality-Grouped Masked Autoencoder for Lunar Foundation Model.

V2 architecture — groups 28 channels into 7 physical modalities:
  surface (4ch), thermal (4ch), spectral_m3 (8ch), gravity (3ch),
  radar (2ch), hapke (4ch), composition (3ch)

Each group shares a multi-channel Conv2d tokenizer that learns intra-group
correlations (e.g., spectral shape across M3 bands). Shared ViT encoder
processes visible tokens from all groups. Shared lightweight decoder with
per-group prediction heads.

~99M params (vs 181M for per-channel V1).

Usage:
    python lunar_mae_v2.py --test-only
"""

import itertools
import math
from dataclasses import dataclass

import torch.distributed

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHANNELS_V4, MODALITY_GROUPS, MODALITY_GROUP_NAMES


# =====================================================================
# ViT components (shared with V1)
# =====================================================================

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # SDPA (FlashAttention when attn_mask=None, math backend otherwise)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# =====================================================================
# Positional embedding utilities
# =====================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """Generate 2D sinusoidal positional embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, -1)  # (2, N)

    half_dim = embed_dim // 2
    emb_h = _get_1d_sincos_pos_embed(half_dim, grid[1])
    emb_w = _get_1d_sincos_pos_embed(half_dim, grid[0])
    return np.concatenate([emb_h, emb_w], axis=1)  # (N, D)


def _get_1d_sincos_pos_embed(embed_dim, pos):
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega = 1.0 / (10000 ** (omega / (embed_dim // 2)))
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


# =====================================================================
# Modality Group definition
# =====================================================================

@dataclass
class ModalityGroupInfo:
    """Metadata for a modality group."""
    name: str
    channel_names: list
    channel_indices: list  # indices into CHANNELS_V4
    n_channels: int


def build_group_info(channel_names: list[str] = None,
                     groups: dict = None) -> list[ModalityGroupInfo]:
    """Build group info from config."""
    if channel_names is None:
        channel_names = CHANNELS_V4
    if groups is None:
        groups = MODALITY_GROUPS

    ch_to_idx = {name: i for i, name in enumerate(channel_names)}
    result = []
    for group_name in MODALITY_GROUP_NAMES:
        group = groups[group_name]
        ch_list = group["channels"]
        indices = [ch_to_idx[ch] for ch in ch_list if ch in ch_to_idx]
        result.append(ModalityGroupInfo(
            name=group_name,
            channel_names=[ch for ch in ch_list if ch in ch_to_idx],
            channel_indices=indices,
            n_channels=len(indices),
        ))
    return result


# =====================================================================
# Grouped MAE Model
# =====================================================================

class LunarMAEv2(nn.Module):
    """Modality-Grouped Masked Autoencoder.

    Groups 28 channels into 7 modality groups, each with a shared
    multi-channel tokenizer. Shared ViT encoder + shared decoder
    with per-group prediction heads.
    """

    def __init__(
        self,
        channel_names: list[str],
        groups: list[ModalityGroupInfo] = None,
        img_size: int = 256,
        patch_size: int = 16,
        # Encoder
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        # Decoder
        decoder_embed_dim: int = 384,
        decoder_depth: int = 4,
        decoder_num_heads: int = 6,
        # Masking
        mask_ratio: float = 0.75,
        # Cross-modal enhancements
        crossmodal_prob: float = 0.0,   # prob of complementary masking
        contrast_weight: float = 0.0,   # weight for contrastive loss
        # Other
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 256
        self.grid_size = img_size // patch_size          # 16
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_ratio = mask_ratio
        self.crossmodal_prob = crossmodal_prob
        self.contrast_weight = contrast_weight

        # Build group info
        if groups is None:
            groups = build_group_info(channel_names)
        self.groups = groups
        self.n_groups = len(groups)
        self.group_names = [g.name for g in groups]

        # Per-group tokenizers: Conv2d(n_channels_in_group, embed_dim, patch_size)
        self.tokenizers = nn.ModuleDict()
        for g in groups:
            self.tokenizers[g.name] = nn.Conv2d(
                g.n_channels, embed_dim,
                kernel_size=patch_size, stride=patch_size)

        # Shared spatial positional embedding (256 patches)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim))

        # Learned modality embedding per group
        self.group_embed = nn.ParameterDict({
            g.name: nn.Parameter(torch.zeros(1, 1, embed_dim))
            for g in groups
        })

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, decoder_embed_dim))

        # Per-group mask tokens (learned)
        self.mask_tokens = nn.ParameterDict({
            g.name: nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            for g in groups
        })

        # Per-group modality embedding for decoder
        self.decoder_group_embed = nn.ParameterDict({
            g.name: nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            for g in groups
        })

        # Cross-modal decoder attention: each group attends to ALL encoder tokens
        self.decoder_cross_attn = nn.MultiheadAttention(
            decoder_embed_dim, num_heads=decoder_num_heads,
            batch_first=True, dropout=drop_rate)
        self.decoder_cross_norm = nn.LayerNorm(decoder_embed_dim)

        # Shared decoder blocks
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, drop=drop_rate)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        # Per-group prediction heads: decoder_dim → n_channels × patch_size²
        self.pred_heads = nn.ModuleDict()
        for g in groups:
            out_dim = g.n_channels * patch_size ** 2
            self.pred_heads[g.name] = nn.Linear(decoder_embed_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        # Sinusoidal pos embeddings
        pos = get_2d_sincos_pos_embed(self.embed_dim, self.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        dec_pos = get_2d_sincos_pos_embed(self.decoder_embed_dim, self.grid_size)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos).float().unsqueeze(0))

        # Xavier for linear/conv, normal for tokens/embeddings
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.cls_token, std=0.02)
        for g in self.groups:
            nn.init.normal_(self.group_embed[g.name], std=0.02)
            nn.init.normal_(self.mask_tokens[g.name], std=0.02)
            nn.init.normal_(self.decoder_group_embed[g.name], std=0.02)

    # -----------------------------------------------------------------
    # Masking
    # -----------------------------------------------------------------

    def random_masking(self, n_patches: int, mask_ratio: float,
                       batch_size: int, device: torch.device):
        """Generate random masks. Returns (B, N) bool where True=VISIBLE."""
        n_keep = max(1, int(n_patches * (1 - mask_ratio)))
        # Create random noise, sort to get indices
        noise = torch.rand(batch_size, n_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        # Keep first n_keep
        mask = torch.zeros(batch_size, n_patches, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :n_keep], True)
        return mask  # True = visible

    # -----------------------------------------------------------------
    # Encoder
    # -----------------------------------------------------------------

    def forward_encoder(self, pixels, has_group, mask_ratio=None):
        """Encode visible tokens from all groups.

        Args:
            pixels: (B, 28, H, W) full input tensor
            has_group: (B, n_groups) bool indicating available groups
            mask_ratio: override mask ratio (None = use self.mask_ratio)

        Returns:
            encoded: (B, max_tokens, D) encoded visible tokens
            group_masks: dict[group_name] → (B, n_patches) bool (True=visible)
            group_token_ranges: dict[group_name] → (start, end) index into encoded
        """
        B = pixels.shape[0]
        device = pixels.device
        has_group = has_group.bool()
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        all_tokens = []           # list of (B, n_visible_i, D) tensors
        all_valid = []            # list of (B, n_visible_i) bool — True=real, False=ghost
        group_masks = {}          # group_name → (B, 256) bool
        group_token_ranges = {}   # group_name → (start, end)

        token_offset = 1  # offset 0 is CLS token

        # Complementary cross-modal masking: with probability crossmodal_prob,
        # keep 1-2 anchor groups fully visible and mask others at 90%.
        # In DDP, broadcast random decisions from rank 0 so all ranks use
        # the same masking strategy for each step.
        use_crossmodal = False
        anchor_groups = set()
        crossmodal_mask_ratio = 0.90
        if self.training and self.crossmodal_prob > 0:
            # Pack random decisions into a tensor for DDP broadcast
            # [0] = use_crossmodal (0/1), [1] = n_anchor, [2:] = anchor indices
            rand_buf = torch.zeros(2 + self.n_groups, device=device, dtype=torch.long)
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                if torch.rand(1).item() < self.crossmodal_prob:
                    rand_buf[0] = 1
                    rand_buf[1] = 1 + torch.randint(2, (1,)).item()  # 1 or 2
                    perm = torch.randperm(self.n_groups)
                    rand_buf[2:] = perm
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(rand_buf, src=0)

            if rand_buf[0].item() == 1:
                # Pick anchors from groups available for at least 1 sample
                available_names = []
                for g in self.groups:
                    g_idx = self.group_names.index(g.name)
                    if has_group[:, g_idx].any():
                        available_names.append(g.name)
                if len(available_names) >= 2:
                    use_crossmodal = True
                    n_anchor = int(rand_buf[1].item())
                    # Use the broadcast permutation to select anchors
                    perm = rand_buf[2:].tolist()
                    # Map permutation indices to available group names
                    all_names = self.group_names
                    selected = []
                    for idx in perm:
                        if all_names[idx] in available_names:
                            selected.append(all_names[idx])
                            if len(selected) >= n_anchor:
                                break
                    anchor_groups = set(selected) if selected else set()

        for g in self.groups:
            g_idx = self.group_names.index(g.name)
            # Check which batch elements have this group
            g_avail = has_group[:, g_idx]  # (B,)

            # Extract group channels: (B, n_ch, H, W)
            ch_indices = torch.tensor(g.channel_indices, device=device)
            group_pixels = pixels[:, ch_indices]  # (B, n_ch, H, W)

            # Tokenize: Conv2d(n_ch, D, P, P) → (B, D, grid, grid) → (B, N, D)
            tokens = self.tokenizers[g.name](group_pixels)
            tokens = tokens.flatten(2).transpose(1, 2)  # (B, 256, D)

            # Add spatial + group positional embeddings
            tokens = tokens + self.pos_embed + self.group_embed[g.name]

            # Generate mask — complementary or standard
            if use_crossmodal and g.name in anchor_groups:
                # Anchor group: fully visible
                mask = torch.ones(B, self.n_patches, dtype=torch.bool, device=device)
            elif use_crossmodal:
                # Non-anchor group: heavily masked (90%)
                mask = self.random_masking(
                    self.n_patches, crossmodal_mask_ratio, B, device)
            else:
                # Standard independent masking
                mask = self.random_masking(
                    self.n_patches, mask_ratio, B, device)  # (B, 256) True=visible

            # Zero out unavailable groups entirely
            mask = mask & g_avail.unsqueeze(1)  # (B, 256)
            group_masks[g.name] = mask

            # Gather visible tokens
            n_visible = mask.sum(dim=1).max().item()  # max across batch
            if n_visible == 0:
                group_token_ranges[g.name] = (token_offset, token_offset)
                continue

            # Pad to uniform length within this group
            # Sort mask indices so visible come first.
            # Cast bool -> int8: ROCm/HIP does not support sort on bool dtype, and
            # argsort of (~mask) as {0,1} gives the identical visible-first ordering.
            ids_sort = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)  # visible first
            visible_tokens = torch.gather(
                tokens, 1,
                ids_sort[:, :n_visible].unsqueeze(-1).expand(-1, -1, self.embed_dim))

            # Track which tokens are real vs ghost (unavailable group)
            # Per-batch: True if group is available AND token was selected as visible
            token_valid = g_avail.unsqueeze(1).expand(-1, n_visible)  # (B, n_visible)

            # Zero out tokens for batch elements without this group
            unavail = ~g_avail  # (B,)
            if unavail.any():
                visible_tokens[unavail] = 0.0

            all_tokens.append(visible_tokens)
            all_valid.append(token_valid)
            group_token_ranges[g.name] = (token_offset, token_offset + n_visible)
            token_offset += n_visible

        # Prepend CLS token (always valid)
        cls = self.cls_token.expand(B, -1, -1)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=device)
        if all_tokens:
            all_tokens = torch.cat(all_tokens, dim=1)  # (B, total_visible, D)
            all_tokens = torch.cat([cls, all_tokens], dim=1)
            all_valid = torch.cat([cls_valid] + all_valid, dim=1)  # (B, 1+total_visible)
        else:
            all_tokens = cls
            all_valid = cls_valid

        # Build attention mask: ghost tokens cannot attend or be attended to
        # valid_mask: (B, N) bool — True=real token
        # SDPA expects (B, 1, N, N) or (B, H, N, N) float mask where -inf = masked
        has_ghost = not all_valid.all()
        attn_mask = None
        if has_ghost:
            # Only mask keys (columns), not queries (rows).
            # Masking query rows would cause softmax([-inf,...,-inf]) = NaN for ghost queries.
            # Ghost queries produce irrelevant outputs but avoid NaN propagation.
            valid_col = all_valid.unsqueeze(1).unsqueeze(2).float()   # (B, 1, 1, N)
            attn_mask = torch.log(valid_col).expand(-1, -1, all_valid.shape[1], -1)  # (B, 1, N, N)

        # Run encoder
        for block in self.encoder_blocks:
            all_tokens = block(all_tokens, attn_mask=attn_mask)
        all_tokens = self.encoder_norm(all_tokens)

        return all_tokens, group_masks, group_token_ranges, all_valid

    # -----------------------------------------------------------------
    # Decoder
    # -----------------------------------------------------------------

    def forward_decoder(self, encoded, group_masks, group_token_ranges,
                        encoder_valid=None, has_group=None):
        """Decode all groups in a single batched pass.

        Instead of running the decoder 7 times sequentially, we stack all
        active groups along the batch dimension: (B*G, 256, D_dec) and run
        the shared decoder once. This maximizes GPU utilization.

        Args:
            encoded: (B, total_vis+1, D) encoder output (with CLS at pos 0)
            group_masks: dict[group_name] → (B, 256) bool
            group_token_ranges: dict[group_name] → (start, end)
            encoder_valid: (B, total_vis+1) bool — True=real token, False=ghost
            has_group: (B, n_groups) bool — which groups are available per sample

        Returns:
            predictions: dict[group_name] → (B, n_patches, n_ch * P²)
        """
        B = encoded.shape[0]
        device = encoded.device
        D = self.decoder_embed_dim
        N = self.n_patches
        if has_group is not None:
            has_group = has_group.bool()

        # Project encoder output to decoder dim
        encoded_dec = self.decoder_proj(encoded)  # (B, total_vis+1, D_dec)

        # Collect active groups and build batched decoder input
        active_groups = []
        group_seqs = []

        for g in self.groups:
            mask = group_masks[g.name]  # (B, 256) True=visible
            start, end = group_token_ranges[g.name]
            n_visible = end - start

            if n_visible == 0 and not mask.any():
                continue

            active_groups.append(g)

            # Build full sequence: start with mask tokens
            mask_token = self.mask_tokens[g.name]
            full_seq = mask_token.expand(B, N, -1).to(encoded_dec.dtype).clone()

            if n_visible > 0:
                group_enc = encoded_dec[:, start:end]  # (B, n_visible, D_dec)
                # bool->int8: ROCm/HIP has no bool sort (same visible-first ordering).
                ids_sort = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)

                # Vectorized scatter: use scatter_ instead of per-batch loop
                # ids_sort[:, :n_visible] gives the original positions of visible tokens
                vis_positions = ids_sort[:, :n_visible]  # (B, n_visible)
                full_seq.scatter_(1,
                    vis_positions.unsqueeze(-1).expand(-1, -1, D),
                    group_enc[:, :n_visible])

                # Restore mask tokens for batch elements where this group is unavailable
                # (scatter wrote zeroed encoder tokens over mask tokens for these)
                if has_group is not None:
                    g_idx = self.group_names.index(g.name)
                    unavail = ~has_group[:, g_idx]  # (B,)
                    if unavail.any():
                        full_seq[unavail] = mask_token.expand(1, N, -1).to(encoded_dec.dtype)

            # Add decoder positional + group embeddings
            full_seq = full_seq + self.decoder_pos_embed + self.decoder_group_embed[g.name]
            group_seqs.append(full_seq)

        if not active_groups:
            return {}

        # Stack all groups: (G*B, 256, D_dec) — single batched decoder pass
        batched = torch.cat(group_seqs, dim=0)  # (G*B, N, D)

        # Cross-modal decoder attention: each group attends to ALL encoder tokens
        # Expand encoder output to match stacked batch: (G*B, N_enc, D_dec)
        G = len(active_groups)
        enc_expanded = encoded_dec.repeat(G, 1, 1)  # (G*B, N_enc, D_dec)

        # Build key_padding_mask to exclude ghost tokens from cross-attention
        # nn.MultiheadAttention key_padding_mask: True = IGNORE
        cross_key_mask = None
        if encoder_valid is not None and not encoder_valid.all():
            cross_key_mask = ~encoder_valid.repeat(G, 1)  # (G*B, N_enc) True=ghost

        normed_batched = self.decoder_cross_norm(batched)
        cross_out, _ = self.decoder_cross_attn(
            normed_batched, enc_expanded, enc_expanded,
            key_padding_mask=cross_key_mask)
        batched = batched + cross_out

        for block in self.decoder_blocks:
            batched = block(batched)
        batched = self.decoder_norm(batched)

        # Split back and apply per-group prediction heads
        predictions = {}
        for i, g in enumerate(active_groups):
            group_out = batched[i * B : (i + 1) * B]  # (B, N, D)
            predictions[g.name] = self.pred_heads[g.name](group_out)

        return predictions

    # -----------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------

    def patchify(self, imgs):
        """Convert (B, C, H, W) → (B, N, C*P*P)."""
        P = self.patch_size
        B, C, H, W = imgs.shape
        h = H // P
        w = W // P
        x = imgs.reshape(B, C, h, P, w, P)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, h, w, C, P, P)
        x = x.reshape(B, h * w, C * P * P)
        return x

    def forward(self, pixels, has_group):
        """Full forward pass with loss computation.

        Args:
            pixels: (B, 28, H, W) input tensor
            has_group: (B, n_groups) bool

        Returns:
            loss: scalar
            metrics: dict with per-group losses
        """
        has_group = has_group.bool()

        # Encode
        encoded, group_masks, group_token_ranges, encoder_valid = self.forward_encoder(
            pixels, has_group)

        # Decode
        predictions = self.forward_decoder(
            encoded, group_masks, group_token_ranges, encoder_valid, has_group)

        # Cross-modal contrastive loss: align group-level representations
        B = pixels.shape[0]
        contrast_loss = torch.tensor(0.0, device=pixels.device)
        if self.contrast_weight > 0:
            # Pool per-group features from encoder output, masking ghost tokens
            group_features = {}
            # Track which batch elements have valid features per group
            group_batch_valid = {}
            for g_name, (start, end) in group_token_ranges.items():
                n_tok = end - start
                if n_tok == 0:
                    continue
                tokens = encoded[:, start:end]  # (B, n_tok, D)
                valid = encoder_valid[:, start:end]  # (B, n_tok) True=real

                # Masked mean pool: only count real tokens
                valid_f = valid.unsqueeze(-1).float()  # (B, n_tok, 1)
                n_valid = valid_f.sum(dim=1).clamp(min=1)  # (B, 1)
                pooled = (tokens * valid_f).sum(dim=1) / n_valid  # (B, D)

                # Track which batch elements have any valid tokens
                batch_has = valid.any(dim=1)  # (B,) bool
                group_features[g_name] = F.normalize(pooled, dim=-1)
                group_batch_valid[g_name] = batch_has

            # InfoNCE between all pairs of available groups
            n_pairs = 0
            feature_names = list(group_features.keys())
            if len(feature_names) >= 2:
                for g1, g2 in itertools.combinations(feature_names, 2):
                    # Only include batch elements where BOTH groups are valid
                    both_valid = group_batch_valid[g1] & group_batch_valid[g2]
                    n_valid_samples = both_valid.sum().item()
                    if n_valid_samples < 2:
                        continue
                    feat1 = group_features[g1][both_valid]  # (K, D)
                    feat2 = group_features[g2][both_valid]  # (K, D)
                    # Cosine similarity matrix in float32 for precision
                    with torch.amp.autocast("cuda", enabled=False):
                        sim = torch.mm(feat1.float(), feat2.float().T) / 0.07
                        labels = torch.arange(n_valid_samples, device=pixels.device)
                        contrast_loss = contrast_loss + (
                            F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
                    n_pairs += 1
                contrast_loss = contrast_loss / max(n_pairs, 1)

        # Compute loss with per-patch validity weighting and inverse-loss group weighting
        group_losses = []  # list of (loss_tensor, name)
        metrics = {}

        for g in self.groups:
            if g.name not in predictions:
                continue

            pred = predictions[g.name]  # (B, 256, n_ch * P²)
            mask = group_masks[g.name]  # (B, 256) True=visible

            # Get target patches
            ch_indices = torch.tensor(g.channel_indices, device=pixels.device)
            group_pixels = pixels[:, ch_indices]  # (B, n_ch, H, W)
            target = self.patchify(group_pixels)  # (B, 256, n_ch * P²)

            # Loss on MASKED patches only (where mask=False means masked)
            g_idx = self.group_names.index(g.name)
            loss_mask = (~mask) & has_group[:, g_idx:g_idx+1]  # (B, 256)

            if loss_mask.sum() == 0:
                continue

            # Per-patch validity: fraction of non-zero pixels in each patch
            # This prevents learning to predict zeros for nodata patches (radar fix)
            valid_frac = (target.abs() > 1e-8).float().mean(dim=-1)  # (B, 256)
            loss_weight = loss_mask.float() * valid_frac  # nodata patches get 0 weight

            # MSE per patch
            per_patch_loss = ((pred - target) ** 2).mean(dim=-1)  # (B, 256)

            denom = loss_weight.sum().clamp(min=1)
            group_loss = (per_patch_loss * loss_weight).sum() / denom

            group_losses.append(group_loss)
            metrics[f"loss_{g.name}"] = group_loss.item()

        # Proportional-loss weighting: groups with higher loss get more weight,
        # so the model focuses on harder modalities. As easy groups converge,
        # hard groups naturally receive more relative gradient.
        if group_losses:
            with torch.no_grad():
                raw_weights = [gl.detach().clamp(min=1e-6) for gl in group_losses]
                w_sum = sum(raw_weights)
                weights = [w / w_sum * len(group_losses) for w in raw_weights]
            recon_loss = sum(w * gl for w, gl in zip(weights, group_losses))
        else:
            recon_loss = torch.tensor(0.0, device=pixels.device)

        # Combine reconstruction + contrastive
        total_loss = recon_loss + self.contrast_weight * contrast_loss

        metrics["loss"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0
        metrics["recon_loss"] = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0
        metrics["contrast_loss"] = contrast_loss.item() if isinstance(contrast_loss, torch.Tensor) else 0.0
        metrics["n_group_losses"] = len(group_losses)

        return total_loss, metrics


# =====================================================================
# Factory functions
# =====================================================================

def lunar_mae_v2_base(channel_names=None, **kwargs):
    """ViT-Base grouped MAE: 768d/12L/12H encoder, 384d/4L/6H decoder."""
    if channel_names is None:
        channel_names = CHANNELS_V4
    return LunarMAEv2(
        channel_names=channel_names,
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=4, decoder_num_heads=6,
        **kwargs,
    )


def lunar_mae_v2_large(channel_names=None, **kwargs):
    """ViT-Large grouped MAE: 1024d/24L/16H encoder, 512d/6L/8H decoder."""
    if channel_names is None:
        channel_names = CHANNELS_V4
    return LunarMAEv2(
        channel_names=channel_names,
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=6, decoder_num_heads=8,
        **kwargs,
    )


# =====================================================================
# Test / CLI
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lunar MAE V2 (Grouped)")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--model", choices=["base", "large"], default="base")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "base":
        model = lunar_mae_v2_base().to(device)
    else:
        model = lunar_mae_v2_large().to(device)

    # Parameter counts
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for n, p in model.named_parameters()
                if "decoder" not in n and "mask_token" not in n
                and "pred_head" not in n)
    n_dec = n_params - n_enc

    print(f"Model: V2 {args.model}")
    print(f"  Groups: {model.n_groups} ({', '.join(g.name for g in model.groups)})")
    print(f"  Channels: {model.n_channels}")
    print(f"  Patches per group: {model.n_patches}")
    print(f"  Total tokens: {model.n_patches * model.n_groups}")
    print(f"  Mask ratio: {model.mask_ratio}")
    print(f"  Total params: {n_params / 1e6:.1f}M")
    print(f"  Encoder params: {n_enc / 1e6:.1f}M")
    print(f"  Decoder params: {n_dec / 1e6:.1f}M")

    # Per-group info
    for g in model.groups:
        tok = model.tokenizers[g.name]
        n_tok_params = sum(p.numel() for p in tok.parameters())
        head = model.pred_heads[g.name]
        n_head_params = sum(p.numel() for p in head.parameters())
        print(f"  {g.name:15s}: {g.n_channels} ch, "
              f"tokenizer={n_tok_params/1e3:.1f}K, head={n_head_params/1e3:.1f}K")

    if args.test_only:
        B = 4
        pixels = torch.randn(B, len(CHANNELS_V4), 256, 256, device=device)
        has_group = torch.ones(B, model.n_groups, dtype=torch.bool, device=device)
        # Simulate radar missing for samples 2,3
        has_group[2:, model.group_names.index("radar")] = False

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss, metrics = model(pixels, has_group)

        print(f"\n  Test forward pass:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Active group losses: {metrics['n_group_losses']}")
        for k, v in sorted(metrics.items()):
            if k.startswith("loss_"):
                print(f"    {k}: {v:.4f}")
