"""Evaluation and visualization for Lunar MAE V2.

Loads a checkpoint, computes per-group reconstruction MSE on fixed samples,
and generates reconstruction visualizations.

Usage:
    python eval_mae.py --checkpoint checkpoints/latest.pt --n-samples 500
    python eval_mae.py --checkpoint checkpoints/latest.pt --visualize --n-vis 8
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import CHANNELS_V4, MODALITY_GROUP_NAMES, MODALITY_GROUPS
from lunar_dataset import MmapRandomCropDataset
from lunar_mae_v2 import LunarMAEv2, lunar_mae_v2_base, lunar_mae_v2_large


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("model_config", {})
    size = config.get("size", "base")
    mask_ratio = config.get("mask_ratio", 0.75)
    patch_size = config.get("patch_size", 16)
    crossmodal_prob = config.get("crossmodal_prob", 0.0)
    contrast_weight = config.get("contrast_weight", 0.0)

    if size == "large":
        model = lunar_mae_v2_large(CHANNELS_V4, mask_ratio=mask_ratio,
                                    patch_size=patch_size,
                                    crossmodal_prob=crossmodal_prob,
                                    contrast_weight=contrast_weight)
    else:
        model = lunar_mae_v2_base(CHANNELS_V4, mask_ratio=mask_ratio,
                                   patch_size=patch_size,
                                   crossmodal_prob=crossmodal_prob,
                                   contrast_weight=contrast_weight)

    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", -1)
    loss = ckpt.get("loss", -1)
    print(f"Loaded checkpoint: epoch={epoch}, loss={loss:.4f}")
    return model, ckpt


@torch.no_grad()
def evaluate(model, dataset, device, n_samples=500, batch_size=32):
    """Compute per-group reconstruction MSE."""
    rng = np.random.RandomState(42)
    n = min(n_samples, len(dataset))

    total_loss = 0.0
    n_batches = 0
    group_losses_sum = {}
    group_losses_count = {}

    batch_pixels = []
    batch_has = []

    for i in range(n):
        sample = dataset[rng.randint(0, len(dataset))]
        batch_pixels.append(sample["pixels"])
        batch_has.append(sample["has_group"])

        if len(batch_pixels) == batch_size or i == n - 1:
            pixels = torch.stack(batch_pixels).to(device)
            has_data = torch.stack(batch_has).to(device).bool()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model(pixels, has_data)

            total_loss += metrics["loss"]
            n_batches += 1

            for k, v in metrics.items():
                if k.startswith("loss_"):
                    group_losses_sum[k] = group_losses_sum.get(k, 0.0) + v
                    group_losses_count[k] = group_losses_count.get(k, 0) + 1

            batch_pixels.clear()
            batch_has.clear()

    avg_loss = total_loss / max(n_batches, 1)
    avg_group = {k: group_losses_sum[k] / group_losses_count[k]
                 for k in group_losses_sum}

    print(f"\nEvaluation ({n} samples):")
    print(f"  Overall MSE: {avg_loss:.6f}")
    for k in sorted(avg_group):
        print(f"  {k}: {avg_group[k]:.6f}")

    return {"overall_mse": avg_loss, **avg_group}


def unpatchify_channel(pred_patches, n_ch, ch_idx, grid, P):
    """Unpatchify one channel from (N_patches, n_ch*P*P) predictions.

    pred_patches: (grid*grid, n_ch * P * P)
    Returns: (grid*P, grid*P) numpy array
    """
    N = grid * grid
    # Reshape to (N, n_ch, P, P)
    patches = pred_patches.reshape(N, n_ch, P, P)
    # Extract target channel
    ch_patches = patches[:, ch_idx]  # (N, P, P)
    # Arrange in grid: (grid, grid, P, P) → (grid*P, grid*P)
    ch_patches = ch_patches.reshape(grid, grid, P, P)
    img = ch_patches.transpose(0, 2, 1, 3).reshape(grid * P, grid * P)
    return img


@torch.no_grad()
def visualize_reconstructions(model, dataset, device, n_vis=4,
                              output_dir="checkpoints"):
    """Generate side-by-side: original | masked input | reconstruction (blend).

    For each group, shows the first channel:
    - Column 1: Original image
    - Column 2: Masked image (visible patches only, masked patches grayed)
    - Column 3: Blended reconstruction (visible = original, masked = predicted)
    """
    output_dir = Path(output_dir)
    rng = np.random.RandomState(123)

    P = model.patch_size
    grid = model.grid_size

    for vis_idx in range(n_vis):
        sample = dataset[rng.randint(0, len(dataset))]
        pixels = sample["pixels"].unsqueeze(0).to(device)
        has_group = sample["has_group"].unsqueeze(0).to(device).bool()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            encoded, group_masks, group_token_ranges, encoder_valid = model.forward_encoder(
                pixels, has_group)
            predictions = model.forward_decoder(
                encoded, group_masks, group_token_ranges, encoder_valid, has_group)

        active_groups = [g for g in model.groups
                         if g.name in predictions and has_group[0, model.group_names.index(g.name)]]

        if not active_groups:
            continue

        n_groups_vis = len(active_groups)
        fig, axes = plt.subplots(n_groups_vis, 3, figsize=(12, 3 * n_groups_vis))
        if n_groups_vis == 1:
            axes = axes[np.newaxis, :]

        for gi, g in enumerate(active_groups):
            pred = predictions[g.name][0].float().cpu().numpy()  # (256, n_ch*P²)
            mask = group_masks[g.name][0].cpu().numpy()  # (256,) True=visible

            # Original first channel
            ch_idx = g.channel_indices[0]
            original = pixels[0, ch_idx].cpu().numpy()  # (256, 256)

            n_ch = g.n_channels
            recon_full = unpatchify_channel(pred, n_ch, 0, grid, P)

            # Build per-pixel mask: True where patch was visible
            mask_2d = mask.reshape(grid, grid)
            mask_pixels = np.repeat(np.repeat(mask_2d, P, axis=0), P, axis=1)

            # Masked input: show original where visible, gray where masked
            masked_input = original.copy()
            gray_val = np.median(original[original != 0]) if (original != 0).any() else 0
            masked_input[mask_pixels == 0] = gray_val

            # Blended: original where visible, reconstruction where masked
            blended = original.copy()
            blended[mask_pixels == 0] = recon_full[mask_pixels == 0]

            # Display range from original
            valid = original[original != 0]
            if len(valid) > 10:
                vmin, vmax = np.percentile(valid, [2, 98])
            else:
                vmin, vmax = -2, 2

            axes[gi, 0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
            axes[gi, 0].set_title(f"{g.name} — original")
            axes[gi, 0].axis("off")

            axes[gi, 1].imshow(masked_input, cmap="gray", vmin=vmin, vmax=vmax)
            axes[gi, 1].set_title(f"masked input (25% visible)")
            axes[gi, 1].axis("off")

            axes[gi, 2].imshow(blended, cmap="gray", vmin=vmin, vmax=vmax)
            axes[gi, 2].set_title(f"reconstruction (masked patches filled)")
            axes[gi, 2].axis("off")

        plt.tight_layout()
        out_path = output_dir / f"recon_vis_{vis_idx:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


@torch.no_grad()
def visualize_crossmodal(model, dataset, device, n_vis=2,
                         output_dir="checkpoints"):
    """Cross-modal reconstruction: fully mask one group, reconstruct from others.

    Encodes all OTHER groups at mask_ratio=0 (fully visible), then manually
    runs the decoder for the hidden group using all mask tokens + cross-attention
    to the encoder output from other groups.
    """
    output_dir = Path(output_dir)
    rng = np.random.RandomState(456)

    P = model.patch_size
    grid = model.grid_size
    N = model.n_patches

    for vis_idx in range(n_vis):
        sample = dataset[rng.randint(0, len(dataset))]
        pixels = sample["pixels"].unsqueeze(0).to(device)
        has_group = sample["has_group"].unsqueeze(0).to(device).bool()

        active_groups = [g for g in model.groups
                         if has_group[0, model.group_names.index(g.name)]]
        if len(active_groups) < 2:
            continue

        n_groups_vis = len(active_groups)
        fig, axes = plt.subplots(n_groups_vis, 3, figsize=(12, 3 * n_groups_vis))
        if n_groups_vis == 1:
            axes = axes[np.newaxis, :]

        for gi, target_group in enumerate(active_groups):
            g_idx = model.group_names.index(target_group.name)

            # Encode all OTHER groups fully visible
            crossmodal_has = has_group.clone()
            crossmodal_has[0, g_idx] = False

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                encoded, group_masks, group_token_ranges, encoder_valid = \
                    model.forward_encoder(pixels, crossmodal_has, mask_ratio=0.0)

                # Manually decode the hidden group:
                # All mask tokens + cross-attend to encoder tokens from other groups
                encoded_dec = model.decoder_proj(encoded)
                mask_token = model.mask_tokens[target_group.name]
                full_seq = mask_token.expand(1, N, -1).to(encoded_dec.dtype).clone()
                full_seq = full_seq + model.decoder_pos_embed + \
                    model.decoder_group_embed[target_group.name]

                # Cross-attention to encoder output
                cross_key_mask = None
                if encoder_valid is not None and not encoder_valid.all():
                    cross_key_mask = ~encoder_valid

                normed = model.decoder_cross_norm(full_seq)
                cross_out, _ = model.decoder_cross_attn(
                    normed, encoded_dec, encoded_dec,
                    key_padding_mask=cross_key_mask)
                full_seq = full_seq + cross_out

                # Shared decoder blocks + prediction head
                for block in model.decoder_blocks:
                    full_seq = block(full_seq)
                full_seq = model.decoder_norm(full_seq)
                pred = model.pred_heads[target_group.name](full_seq)

            # Original first channel
            ch_idx = target_group.channel_indices[0]
            original = pixels[0, ch_idx].cpu().numpy()

            # Reconstruction
            pred_np = pred[0].float().cpu().numpy()
            recon = unpatchify_channel(pred_np, target_group.n_channels, 0, grid, P)

            # Difference
            diff = np.abs(original - recon)

            # Display range
            valid = original[original != 0]
            if len(valid) > 10:
                vmin, vmax = np.percentile(valid, [2, 98])
            else:
                vmin, vmax = -2, 2

            axes[gi, 0].imshow(original, cmap="gray", vmin=vmin, vmax=vmax)
            axes[gi, 0].set_title(f"{target_group.name} — original")
            axes[gi, 0].axis("off")

            axes[gi, 1].imshow(recon, cmap="gray", vmin=vmin, vmax=vmax)
            axes[gi, 1].set_title(f"cross-modal recon (group hidden)")
            axes[gi, 1].axis("off")

            axes[gi, 2].imshow(diff, cmap="hot", vmin=0, vmax=(vmax - vmin) * 0.3)
            axes[gi, 2].set_title(f"|error|")
            axes[gi, 2].axis("off")

        plt.suptitle("Cross-Modal Reconstruction (each group fully masked, "
                     "reconstructed from others)", fontsize=11, y=1.01)
        plt.tight_layout()
        out_path = output_dir / f"crossmodal_vis_{vis_idx:02d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Lunar MAE V2 Evaluation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--crossmodal", action="store_true",
                        help="Generate cross-modal reconstruction visualizations")
    parser.add_argument("--n-vis", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--test-set", action="store_true",
                        help="Use fixed test split patches instead of random crops")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)

    if args.test_set:
        from downstream_dataset import MmapPatchDataset
        dataset = MmapPatchDataset(split=2, channel_names=CHANNELS_V4)
        print(f"Using test split: {len(dataset)} patches")
    else:
        dataset = MmapRandomCropDataset(
            channel_names=CHANNELS_V4,
            crop_size=256,
            epoch_length=10_000,
            min_valid_fraction=0.3,
        )

    results = evaluate(model, dataset, device, n_samples=args.n_samples)

    # Save results
    suffix = "_test" if args.test_set else ""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_results{suffix}.json"
    epoch = ckpt.get("epoch", -1)
    results["epoch"] = epoch
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if args.visualize:
        print("\nGenerating reconstruction visualizations...")
        visualize_reconstructions(model, dataset, device,
                                  n_vis=args.n_vis,
                                  output_dir=args.output_dir)

    if args.crossmodal:
        print("\nGenerating cross-modal reconstruction visualizations...")
        visualize_crossmodal(model, dataset, device,
                             n_vis=args.n_vis,
                             output_dir=args.output_dir)

    dataset.close()


if __name__ == "__main__":
    main()
