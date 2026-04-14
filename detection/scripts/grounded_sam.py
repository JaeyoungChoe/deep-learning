"""Grounded-SAM: Grounding DINO + SAM for text-prompted segmentation.

Usage:
    python scripts/grounded_sam.py <image> "query1. query2." [--threshold 0.3]
    python scripts/grounded_sam.py inputs/sample.jpg "monitor. keyboard. chair."
    python scripts/grounded_sam.py inputs/sample.jpg "person. laptop." --dino-model base --sam-model sam2
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
    Sam2Model,
    Sam2Processor,
)

COLORS = [
    (255, 56, 56), (72, 249, 10), (0, 152, 255), (255, 178, 29),
    (187, 85, 255), (46, 243, 171), (255, 112, 31), (27, 231, 255),
    (89, 61, 246), (207, 210, 49), (255, 56, 224), (255, 157, 151),
]

DINO_MODELS = {
    "tiny": "IDEA-Research/grounding-dino-tiny",
    "base": "IDEA-Research/grounding-dino-base",
}

SAM_MODELS = {
    "sam-base": "facebook/sam-vit-base",
    "sam-large": "facebook/sam-vit-large",
    "sam-huge": "facebook/sam-vit-huge",
    "sam2": "facebook/sam2-hiera-large",
}


def overlay_mask(image, mask, color, alpha=0.45):
    """Overlay a single binary mask on image."""
    overlay = np.array(image, dtype=np.float32)
    mask_bool = mask.astype(bool)
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
            overlay[:, :, c],
        )
    return Image.fromarray(overlay.astype(np.uint8))


def draw_results(image, boxes, scores, labels, masks, color_map):
    """Draw masks, boxes, and labels on image."""
    result = image.copy()

    # Overlay masks
    for mask, label in zip(masks, labels):
        color = color_map.get(label, COLORS[0])
        result = overlay_mask(result, mask, color)

    # Draw mask borders
    for mask, label in zip(masks, labels):
        color = color_map.get(label, COLORS[0])
        contour = np.zeros_like(mask, dtype=np.uint8)
        m = mask.astype(np.uint8)
        # Simple edge detection: dilate - original
        from scipy.ndimage import binary_dilation
        dilated = binary_dilation(m, iterations=2).astype(np.uint8)
        contour = dilated - m
        contour_img = np.array(result)
        contour_img[contour.astype(bool)] = color
        result = Image.fromarray(contour_img)

    # Draw boxes and labels
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(c, 1) for c in box.tolist()]
        color = color_map.get(label, COLORS[0])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        text = f"{label} {score:.0%}"
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 4), text, fill="white", font=font)

    return result


def main():
    parser = argparse.ArgumentParser(description="Grounded-SAM: text-prompted segmentation")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("queries", help='Period-separated queries (e.g. "cat. dog. person.")')
    parser.add_argument("--threshold", type=float, default=0.3, help="Box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="Text confidence threshold")
    parser.add_argument("--dino-model", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--sam-model", choices=list(SAM_MODELS.keys()), default="sam-base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = args.device

    # Parse queries: "monitor. keyboard." → ["monitor", "keyboard"]
    queries = [q.strip().rstrip(".") for q in args.queries.split(".") if q.strip()]
    # Grounding DINO expects period-separated text
    text_prompt = ". ".join(queries) + "."
    text_labels = [queries]
    print(f"Queries: {queries}")
    print(f"Text prompt: {text_prompt!r}")

    # --- Step 1: Grounding DINO ---
    dino_id = DINO_MODELS[args.dino_model]
    print(f"\n[1/2] Loading Grounding DINO ({dino_id})...")
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device)
    dino_model.eval()

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.width}x{image.height}")

    inputs = dino_processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        threshold=args.threshold,
        text_threshold=args.text_threshold,
        target_sizes=[(image.height, image.width)],
    )

    result = results[0]
    boxes = result["boxes"]  # (N, 4)
    scores = result["scores"]
    det_labels = result["text_labels"]

    print(f"Detected {len(boxes)} objects:")
    for box, score, label in zip(boxes, scores, det_labels):
        coords = [round(c, 1) for c in box.tolist()]
        print(f"  {label}: {score:.1%} at {coords}")

    if len(boxes) == 0:
        print("No objects detected. Try lowering --threshold.")
        return

    # --- Step 2: SAM segmentation ---
    sam_id = SAM_MODELS[args.sam_model]
    print(f"\n[2/2] Loading SAM ({sam_id})...")

    is_sam2 = "sam2" in args.sam_model
    if is_sam2:
        sam_processor = Sam2Processor.from_pretrained(sam_id)
        sam_model = Sam2Model.from_pretrained(sam_id).to(device)
    else:
        sam_processor = SamProcessor.from_pretrained(sam_id)
        sam_model = SamModel.from_pretrained(sam_id).to(device)
    sam_model.eval()

    # Prepare boxes for SAM: list of list of boxes per image
    input_boxes = [boxes.cpu().tolist()]
    sam_inputs = sam_processor(images=image, input_boxes=input_boxes, return_tensors="pt").to(device)

    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.post_process_masks(
        sam_outputs.pred_masks,
        sam_inputs["original_sizes"],
        sam_inputs["reshaped_input_sizes"],
    )

    # masks[0] shape: (num_boxes, num_predictions_per_box, H, W)
    # Take the best mask for each box (highest IoU prediction = index 0 usually works)
    masks_np = masks[0].cpu().numpy()
    if masks_np.ndim == 4:
        # Select best mask per box (index with highest predicted IoU)
        if hasattr(sam_outputs, "iou_scores"):
            iou = sam_outputs.iou_scores[0].cpu().numpy()  # (num_boxes, num_preds)
            best_idx = iou.argmax(axis=1)
            masks_np = np.array([masks_np[i, best_idx[i]] for i in range(len(best_idx))])
        else:
            masks_np = masks_np[:, 0]  # fallback: take first prediction

    print(f"Generated {len(masks_np)} segmentation masks")

    # --- Draw and save ---
    color_map = {q: COLORS[i % len(COLORS)] for i, q in enumerate(queries)}
    result_image = draw_results(
        image, boxes.cpu(), scores.cpu(), det_labels, masks_np, color_map,
    )

    out_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(out_dir, f"{stem}_grounded_sam.jpg")
    result_image.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
