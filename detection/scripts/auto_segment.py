"""SAM Automatic Mask Generator - no text prompt needed.

Segments ALL objects in an image automatically.

Usage:
    python scripts/auto_segment.py <image> [--model sam-base|sam-large|sam-huge]
    python scripts/auto_segment.py inputs/sample.jpg
    python scripts/auto_segment.py inputs/sample.jpg --model sam-huge --points-per-side 48
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import SamModel, SamProcessor, pipeline

SAM_MODELS = {
    "sam-base": "facebook/sam-vit-base",
    "sam-large": "facebook/sam-vit-large",
    "sam-huge": "facebook/sam-vit-huge",
}

# Distinct color palette (20 colors)
COLORS = [
    (255, 56, 56), (72, 249, 10), (0, 152, 255), (255, 178, 29),
    (187, 85, 255), (46, 243, 171), (255, 112, 31), (27, 231, 255),
    (89, 61, 246), (207, 210, 49), (255, 56, 224), (255, 157, 151),
    (128, 0, 0), (0, 128, 128), (128, 128, 0), (64, 0, 128),
    (255, 200, 150), (100, 200, 100), (200, 100, 200), (50, 150, 255),
]


def create_colorful_overlay(image, masks, scores, alpha=0.45):
    """Overlay all masks with distinct colors, sorted by area (largest first)."""
    overlay = np.array(image, dtype=np.float32)
    h, w = overlay.shape[:2]

    # Sort by area descending (draw large masks first, small on top)
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]

    for rank, idx in enumerate(order):
        mask = masks[idx].astype(bool)
        color = COLORS[rank % len(COLORS)]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )

    # Draw contours
    from scipy.ndimage import binary_dilation
    result = overlay.astype(np.uint8)
    for rank, idx in enumerate(order):
        mask = masks[idx].astype(np.uint8)
        dilated = binary_dilation(mask, iterations=2).astype(np.uint8)
        contour = (dilated - mask).astype(bool)
        color = COLORS[rank % len(COLORS)]
        result[contour] = color

    return Image.fromarray(result)


def main():
    parser = argparse.ArgumentParser(description="SAM auto segmentation (no prompt)")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", choices=list(SAM_MODELS.keys()), default="sam-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points-per-side", type=int, default=32,
                        help="Grid density for auto mask generation (default: 32)")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88,
                        help="Filter masks below this IoU score (default: 0.88)")
    parser.add_argument("--stability-score-thresh", type=float, default=0.95,
                        help="Filter masks below this stability score (default: 0.95)")
    parser.add_argument("--min-mask-area", type=int, default=100,
                        help="Minimum mask area in pixels (default: 100)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    model_id = SAM_MODELS[args.model]
    print(f"Model: {model_id}")
    print(f"Points per side: {args.points_per_side}")

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.width}x{image.height}")

    # Use transformers pipeline for automatic mask generation
    print("Loading model & generating masks...")
    generator = pipeline(
        "mask-generation",
        model=model_id,
        device=args.device,
        points_per_batch=64,
    )

    outputs = generator(
        image,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )

    masks = outputs["masks"]  # list of 2D numpy arrays
    scores = outputs["scores"]

    # Filter by min area
    filtered = [(m, s) for m, s in zip(masks, scores)
                if np.array(m).sum() >= args.min_mask_area]
    if filtered:
        masks, scores = zip(*filtered)
        masks, scores = list(masks), list(scores)
    else:
        masks, scores = [], []

    masks = [np.array(m) for m in masks]
    print(f"Found {len(masks)} segments")

    if len(masks) == 0:
        print("No masks generated. Try lowering thresholds.")
        return

    # Sort by score for reporting
    for i, (m, s) in enumerate(sorted(zip(masks, scores), key=lambda x: -x[1])[:10]):
        area_pct = 100 * m.sum() / (image.width * image.height)
        print(f"  Segment {i+1}: score={s:.3f}, area={area_pct:.1f}%")
    if len(masks) > 10:
        print(f"  ... and {len(masks) - 10} more")

    # Create visualization
    result_image = create_colorful_overlay(image, masks, scores)

    # Add count label
    draw = ImageDraw.Draw(result_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    text = f"{len(masks)} segments"
    draw.rectangle([8, 8, 180, 36], fill=(0, 0, 0, 180))
    draw.text((12, 10), text, fill="white", font=font)

    # Save
    out_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(out_dir, f"{stem}_auto_seg.jpg")
    result_image.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
