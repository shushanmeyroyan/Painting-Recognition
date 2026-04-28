#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO segmentation cropper on synthetic painting images.")
    parser.add_argument("--data", default="data/synthetic_yolo_paintings/data.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Use yolo11n-seg.pt or yolov8n-seg.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--project", default="data/yolo_runs")
    parser.add_argument("--name", default="painting_cropper")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install ultralytics first: pip install ultralytics") from exc

    model = YOLO(args.model)
    model.train(
        data=str(Path(args.data)),
        epochs=args.epochs,
        imgsz=args.imgsz,
        task="segment",
        project=args.project,
        name=args.name,
    )
    print(f"Best weights should be under {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    print("Copy/rename best.pt to data/models/painting_yolo_seg.pt for the app cropper.")


if __name__ == "__main__":
    main()
