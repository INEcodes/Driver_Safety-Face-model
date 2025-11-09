r"""
Graph.py - Face-fit analysis using YOLO (ultralytics)
Scans Dataset2 subfolders (classes), runs a YOLO face model on each image,
saves annotated images and a per-class bar chart showing percent of images
where the largest face is near the image center ("fit").

Usage (example):
    python Graph.py --dataset "c:/Abhay D Drive/Coding/4th year Capstone project/Dataset2" \
        --model "yolov8n-face.pt" --out "outputs"
"""
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError("Install ultralytics: pip install ultralytics") from e


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def annotate_and_save(img, boxes, labels, out_path):
    """Draw bounding boxes and labels on image and save."""
    annotated = img.copy()
    for (x1, y1, x2, y2), text in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
        (label_w, label_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 200, 0), -1)
        cv2.putText(annotated, text, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), annotated)


def process_dataset(dataset_dir: str, model_path: str = "yolov8n.pt",
                    conf_thresh: float = 0.25, iou: float = 0.45,
                    center_threshold: float = 0.18, output_dir: str = "outputs"):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    ensure_dir(output_dir)
    annotated_dir = output_dir / "annotated"
    ensure_dir(annotated_dir)

    try:
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model '{model_path}': {e}") from e

    class_stats = defaultdict(lambda: {"fit": 0, "total": 0})

    class_dirs = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        print(f"Warning: No class directories found in {dataset_dir}")
        return class_stats, None

    print(f"Found {len(class_dirs)} classes: {[d.name for d in class_dirs]}")

    for class_dir in class_dirs:
        images = [p for p in class_dir.rglob("*")
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        if not images:
            print(f"Warning: No images found in {class_dir.name}")
            continue

        pbar = tqdm(images, desc=f"Processing {class_dir.name}", unit="img")
        for img_path in pbar:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    pbar.write(f"Warning: Could not read {img_path.name}")
                    continue

                h, w = img.shape[:2]

                # inference (handle different ultralytics call styles)
                results = None
                try:
                    # prefer model.predict if available
                    results = model.predict(source=str(img_path), conf=conf_thresh, iou=iou, verbose=False,
                                             device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu')
                except Exception:
                    try:
                        results = model(str(img_path), conf=conf_thresh, iou=iou, verbose=False)
                    except Exception as e:
                        pbar.write(f"Error running model on {img_path.name}: {e}")
                        continue

                # normalize results to a single result object
                res = results[0] if isinstance(results, (list, tuple)) and len(results) > 0 else results

                # extract boxes and confidences robustly across API versions
                xyxy = np.array([])
                confs = np.array([])
                try:
                    boxes_obj = getattr(res, "boxes", None)
                    if boxes_obj is not None:
                        # xyxy
                        if hasattr(boxes_obj, "xyxy"):
                            xyxy_attr = boxes_obj.xyxy
                            # handle torch tensor or numpy array
                            if hasattr(xyxy_attr, "cpu"):
                                xyxy = xyxy_attr.cpu().numpy()
                            else:
                                xyxy = np.array(xyxy_attr)
                        # confidences
                        if hasattr(boxes_obj, "conf"):
                            conf_attr = boxes_obj.conf
                            if hasattr(conf_attr, "cpu"):
                                confs = conf_attr.cpu().numpy()
                            else:
                                confs = np.array(conf_attr)
                except Exception as e:
                    pbar.write(f"Warning: Error extracting boxes from {img_path.name}: {e}")
                    xyxy = np.array([])
                    confs = np.array([])

                boxes = []
                scores = []
                if xyxy.size > 0:
                    for b, c in zip(xyxy, confs):
                        boxes.append(b.tolist())
                        scores.append(float(c))

                # determine face-fit for largest box
                is_fit = False
                labels = []
                if boxes:
                    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
                    idx = int(np.argmax(areas))
                    main_box = boxes[idx]
                    cx = (main_box[0] + main_box[2]) / (2.0 * w)
                    cy = (main_box[1] + main_box[3]) / (2.0 * h)
                    dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
                    is_fit = dist < center_threshold

                    for i, (b, score) in enumerate(zip(boxes, scores)):
                        tag = f"face {score:.2f}"
                        if i == idx:
                            tag += " (main)"
                        labels.append(tag)

                # save annotated image
                out_class_dir = annotated_dir / class_dir.name
                ensure_dir(out_class_dir)
                out_file = out_class_dir / img_path.name

                if boxes:
                    annotate_and_save(img, boxes, labels, str(out_file))
                else:
                    cv2.imwrite(str(out_file), img)

                class_stats[class_dir.name]["total"] += 1
                if is_fit:
                    class_stats[class_dir.name]["fit"] += 1

            except Exception as e:
                pbar.write(f"Error processing {img_path.name}: {e}")
                traceback.print_exc()
                continue

        pbar.close()

    if not class_stats:
        print("Warning: No images were processed successfully")
        return class_stats, None

    # prepare chart
    classes = []
    fit_perc = []
    for cls in sorted(class_stats.keys()):
        st = class_stats[cls]
        total = st["total"]
        fit = st["fit"]
        perc = (fit / total * 100) if total > 0 else 0.0
        classes.append(cls)
        fit_perc.append(perc)

    plt.figure(figsize=(max(10, len(classes) * 1.5), 6))
    bars = plt.bar(classes, fit_perc, color="tab:blue", edgecolor='black', linewidth=1.2)
    plt.ylim(0, 105)
    plt.ylabel("Face-fit Percentage (%)", fontsize=12)
    plt.xlabel("Class", fontsize=12)
    plt.title("Face-fit (Centered Face) per Class", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, fit_perc):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    chart_path = output_dir / "face_fit_scores.png"
    plt.tight_layout()
    plt.savefig(str(chart_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"Annotated images saved to: {annotated_dir}")
    print(f"Chart saved to: {chart_path}")
    print(f"{'='*60}\n")

    return class_stats, chart_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute face-fit scores with YOLO on Dataset2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # default to your workspace Dataset2 so script can be run from VS Code without providing --dataset
    default_dataset = r"c:\Abhay D Drive\Coding\4th year Capstone project\Dataset2"
    parser.add_argument("--dataset", "-d", default=default_dataset,
                        help=f"Dataset2 root directory (default: {default_dataset})")
    parser.add_argument("--model", "-m", default="yolov8n.pt",
                        help="YOLO model path (face-capable model recommended)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--center-threshold", type=float, default=0.18,
                        help="Normalized distance threshold for 'fit' (default: 0.18)")
    parser.add_argument("--out", "-o", default="outputs",
                        help="Output directory (default: outputs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"\n{'='*60}")
    print("Face-Fit Analysis Starting")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Confidence: {args.conf}")
    print(f"IoU: {args.iou}")
    print(f"Center Threshold: {args.center_threshold}")
    print(f"Output: {args.out}")
    print(f"{'='*60}\n")

    try:
        stats, chart = process_dataset(
            args.dataset,
            model_path=args.model,
            conf_thresh=args.conf,
            iou=args.iou,
            center_threshold=args.center_threshold,
            output_dir=args.out
        )

        print("\nSummary by Class:")
        print(f"{'='*60}")
        for cls in sorted(stats.keys()):
            st = stats[cls]
            total = st["total"]
            fit = st["fit"]
            pct = (fit / total * 100) if total > 0 else 0.0
            print(f"{cls:20s}: {fit:4d}/{total:4d} fit ({pct:5.1f}%)")
        print(f"{'='*60}\n")
        print("✓ Processing complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()
        exit(1)