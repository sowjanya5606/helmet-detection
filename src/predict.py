from pathlib import Path
import argparse

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "yolov8n": PROJECT_ROOT / "models" / "yolov8n" / "best.pt",
    "yolo11n": PROJECT_ROOT / "models" / "yolo11n" / "best.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Helmet detection using Ultralytics YOLO models."
    )

    parser.add_argument(
        "--model",
        choices=MODELS.keys(),
        required=True,
        help="Model to use for detection.",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image, folder, video, or webcam source.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device.",
    )

    parser.add_argument(
        "--output",
        default="results/predictions/inference",
        help="Directory where prediction results are saved.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_path = MODELS[args.model]
    source = Path(args.source)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not source.exists():
        raise FileNotFoundError(
            f"Source not found: {source}"
        )

    output_dir = PROJECT_ROOT / args.output

    print(f"Model: {args.model}")
    print(f"Weights: {model_path}")
    print(f"Source: {source}")
    print(f"Confidence: {args.conf}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")

    model = YOLO(str(model_path))

    model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(output_dir),
        name=args.model,
        exist_ok=True,
    )

    print("\nPrediction completed.")
    print(f"Results saved under: {output_dir / args.model}")


if __name__ == "__main__":
    main()