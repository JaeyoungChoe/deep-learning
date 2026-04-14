"""OWLv2 zero-shot object detection.

Usage:
    python scripts/detect.py <image_path> "query1, query2, ..." [--threshold 0.3] [--model base|large]
    python scripts/detect.py inputs/sample.jpg "monitor, keyboard, mouse"
    python scripts/detect.py inputs/sample.jpg "사람, 의자" --threshold 0.2
"""
import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Color palette for bounding boxes
COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (46, 243, 171), (27, 231, 255),
    (0, 152, 255), (89, 61, 246), (187, 85, 255), (255, 56, 224),
]

MODEL_MAP = {
    "base": "google/owlv2-base-patch16-ensemble",
    "large": "google/owlv2-large-patch14-ensemble",
}


def draw_boxes(image, boxes, scores, labels, text_queries):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    query_set = sorted(set(text_queries))
    color_map = {q: COLORS[i % len(COLORS)] for i, q in enumerate(query_set)}

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(c, 1) for c in box.tolist()]
        color = color_map.get(label, COLORS[0])

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label background
        text = f"{label} {score:.0%}"
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 6, x1 + tw + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - th - 4), text, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser(description="OWLv2 zero-shot object detection")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("queries", help='Comma-separated text queries (e.g. "cat, dog, person")')
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--model", choices=["base", "large"], default="base", help="Model size (default: base)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: outputs/)")
    args = parser.parse_args()

    # Parse queries
    text_queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    print(f"Queries: {text_queries}")
    print(f"Threshold: {args.threshold}")

    # Load model
    model_name = MODEL_MAP[args.model]
    print(f"Loading {model_name}...")
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(args.device)
    model.eval()
    print("Model loaded.")

    # Load image
    image = Image.open(args.image).convert("RGB")
    print(f"Image: {image.width}x{image.height}")

    # Run inference
    text_labels = [["a photo of a " + q for q in text_queries]]
    inputs = processor(text=text_labels, images=image, return_tensors="pt").to(args.device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)], device=args.device)
    results = processor.post_process_grounded_object_detection(
        outputs=outputs, target_sizes=target_sizes, threshold=args.threshold, text_labels=text_labels,
    )

    result = results[0]
    boxes, scores, labels = result["boxes"], result["scores"], result["text_labels"]
    # Clean label prefix
    labels = [l.replace("a photo of a ", "") for l in labels]

    print(f"\nDetected {len(boxes)} objects:")
    for box, score, label in zip(boxes, scores, labels):
        coords = [round(c, 1) for c in box.tolist()]
        print(f"  {label}: {score:.1%} at {coords}")

    # Draw and save
    result_image = draw_boxes(image.copy(), boxes, scores, labels, text_queries)

    out_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(out_dir, f"{stem}_detected.jpg")
    result_image.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
