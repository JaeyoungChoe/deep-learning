"""OWLv2 zero-shot object detection on video.

Usage:
    python scripts/detect_video.py <video_path> "query1, query2, ..." [--threshold 0.3] [--fps 5]
    python scripts/detect_video.py inputs/sample.mp4 "monitor, keyboard, person" --fps 10
"""
import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import imageio.v2 as imageio

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
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    query_set = sorted(set(text_queries))
    color_map = {q: COLORS[i % len(COLORS)] for i, q in enumerate(query_set)}

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [round(c, 1) for c in box.tolist()]
        color = color_map.get(label, COLORS[0])
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label} {score:.0%}"
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - th - 3), text, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser(description="OWLv2 video object detection")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("queries", help='Comma-separated queries (e.g. "monitor, person")')
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--model", choices=["base", "large"], default="base")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=5, help="Sampling FPS (default: 5)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    text_queries = [q.strip() for q in args.queries.split(",") if q.strip()]
    print(f"Queries: {text_queries}")

    # Load model
    model_name = MODEL_MAP[args.model]
    print(f"Loading {model_name}...")
    processor = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(args.device)
    model.eval()

    # Extract frames
    cap = cv2.VideoCapture(args.video)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, int(video_fps / args.fps))
    print(f"Video: {video_fps:.0f}fps, {total_frames} frames, sampling every {interval} frames")

    text_labels = [["a photo of a " + q for q in text_queries]]
    result_frames = []
    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            inputs = processor(text=text_labels, images=pil_image, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.tensor([(pil_image.height, pil_image.width)], device=args.device)
            results = processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes,
                threshold=args.threshold, text_labels=text_labels,
            )
            result = results[0]
            labels_clean = [l.replace("a photo of a ", "") for l in result["text_labels"]]

            drawn = draw_boxes(pil_image.copy(), result["boxes"], result["scores"], labels_clean, text_queries)
            result_frames.append(np.array(drawn))
            processed += 1

            if processed % 10 == 0:
                print(f"  Processed {processed} frames...")

        frame_idx += 1

    cap.release()
    print(f"Processed {processed} frames total.")

    # Save as GIF and MP4
    out_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.video))[0]

    gif_path = os.path.join(out_dir, f"{stem}_detected.gif")
    imageio.mimsave(gif_path, result_frames, fps=min(args.fps, 15), loop=0)
    print(f"GIF saved: {gif_path} ({os.path.getsize(gif_path)/1024/1024:.1f} MB)")

    mp4_path = os.path.join(out_dir, f"{stem}_detected.mp4")
    writer = imageio.get_writer(mp4_path, fps=min(args.fps, 15))
    for f in result_frames:
        writer.append_data(f)
    writer.close()
    print(f"MP4 saved: {mp4_path} ({os.path.getsize(mp4_path)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
