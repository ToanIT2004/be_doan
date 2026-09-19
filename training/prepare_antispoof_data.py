"""Detect/crop faces from already subject-disjoint raw dataset splits."""

import argparse
import os
from pathlib import Path

import cv2

# Keep Ultralytics settings inside the project on restricted Windows accounts.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("Ultralytics").resolve()))

from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/antispoof_raw"))
    parser.add_argument("--output", type=Path, default=Path("data/antispoof"))
    parser.add_argument("--detector", type=Path, default=Path("models/yolov8n-face.pt"))
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--padding", type=float, default=0.30)
    parser.add_argument("--confidence", type=float, default=0.50)
    return parser.parse_args()


def crop_one_face(image, model, confidence: float, padding: float):
    result = model.predict(image, conf=confidence, imgsz=640, verbose=False)[0]
    rows = result.boxes.data.cpu().tolist() if result.boxes is not None else []
    if len(rows) != 1:
        return None
    x1, y1, x2, y2 = rows[0][:4]
    height, width = image.shape[:2]
    face_width = x2 - x1
    face_height = y2 - y1
    left = max(0, round(x1 - face_width * padding))
    top = max(0, round(y1 - face_height * padding))
    right = min(width, round(x2 + face_width * padding))
    bottom = min(height, round(y2 + face_height * padding))
    crop = image[top:bottom, left:right]
    return crop if crop.size else None


def safe_stem(relative_path: Path) -> str:
    parts = relative_path.with_suffix("").parts
    # Structured input commonly uses subject_id/subject_id__sample.ext. Avoid
    # duplicating the subject prefix in every extracted frame name.
    if len(parts) >= 2 and parts[-1].startswith(f"{parts[-2]}__"):
        return parts[-1].replace(" ", "_")
    return "__".join(parts).replace(" ", "_")


def save_crop(crop, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(target), crop))


def main() -> None:
    args = parse_args()
    if args.sample_every < 1:
        raise ValueError("sample-every must be positive")
    model = YOLO(str(args.detector))
    written = skipped = 0

    for split in ("train", "val", "test"):
        for label in ("live", "spoof"):
            source_root = args.input / split / label
            if not source_root.is_dir():
                if split == "test":
                    continue
                raise FileNotFoundError(f"Missing raw data directory: {source_root}")
            for source in source_root.rglob("*"):
                if not source.is_file():
                    continue
                suffix = source.suffix.lower()
                relative = source.relative_to(source_root)
                stem = safe_stem(relative)
                if suffix in IMAGE_SUFFIXES:
                    image = cv2.imread(str(source))
                    crop = None if image is None else crop_one_face(
                        image, model, args.confidence, args.padding
                    )
                    if crop is None:
                        skipped += 1
                        continue
                    written += int(
                        save_crop(crop, args.output / split / label / f"{stem}.jpg")
                    )
                elif suffix in VIDEO_SUFFIXES:
                    capture = cv2.VideoCapture(str(source))
                    frame_number = 0
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        if frame_number % args.sample_every == 0:
                            crop = crop_one_face(
                                frame, model, args.confidence, args.padding
                            )
                            if crop is None:
                                skipped += 1
                            else:
                                target = (
                                    args.output
                                    / split
                                    / label
                                    / f"{stem}_{frame_number:06d}.jpg"
                                )
                                written += int(save_crop(crop, target))
                        frame_number += 1
                    capture.release()

    print(f"written={written} skipped_no_single_face={skipped}")


if __name__ == "__main__":
    main()
