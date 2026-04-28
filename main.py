#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py build-index [options]")
        print("  python main.py query <image_path> [options]")
        print("  python main.py generate-yolo [options]")
        print("  python main.py train-cropper [options]")
        print("  python main.py export-manifest [options]")
        print("  python main.py evaluate [options]")
        sys.exit(1)

    command = sys.argv[1]
    remaining_args = sys.argv[2:]

    if command == "build-index":
        script = "scripts/build_index.py"
    elif command == "query":
        script = "scripts/query_index.py"
    elif command == "generate-yolo":
        script = "scripts/generate_synthetic_yolo.py"
    elif command == "train-cropper":
        script = "scripts/train_cropper.py"
    elif command == "export-manifest":
        script = "scripts/export_index_manifest.py"
    elif command == "evaluate":
        script = "scripts/evaluate_models.py"
    else:
        print("Unknown command. Use 'build-index', 'query', 'generate-yolo', 'train-cropper', 'export-manifest', or 'evaluate'.")
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    completed = subprocess.run([sys.executable, script, *remaining_args], env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
