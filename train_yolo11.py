# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO11 (Nano/Small) on VisDrone (single 'person' class) on Windows, tracked in W&B.

Usage (PowerShell, from the repo root) — 4 runs: {nano, small} x {base, augmented}:
    python train_yolo11.py --model yolo11n.pt --data data\visdrone_base.yaml       --name nano_base
    python train_yolo11.py --model yolo11n.pt --data data\visdrone_augmented.yaml  --name nano_augmented
    python train_yolo11.py --model yolo11s.pt --data data\visdrone_base.yaml       --name small_base
    python train_yolo11.py --model yolo11s.pt --data data\visdrone_augmented.yaml  --name small_augmented

All 4 commands pass identical --epochs/--imgsz/--batch/--workers so the 4 runs stay comparable;
only --model and --data (and therefore --name) differ. All 4 runs report to the SAME W&B project
(--project / WANDB_PROJECT, e.g. an existing "YOLOv11" project) and are told apart by --name, not
by separate auto-created projects. See README_EXPERIMENTS.md.
"""

import argparse
import os

import torch
from dotenv import load_dotenv


def parse_args():
    """Parse command-line arguments for a single training run."""
    parser = argparse.ArgumentParser(description="Train YOLO11 Nano/Small on VisDrone (Windows, RTX 5060 Ti, 16GB VRAM)")
    parser.add_argument("--data", type=str, required=True, help="Path to visdrone_base.yaml or visdrone_augmented.yaml")
    parser.add_argument(
        "--model", type=str, default="yolo11n.pt", help="Checkpoint to fine-tune from: yolo11n.pt (Nano) or yolo11s.pt (Small)"
    )
    parser.add_argument(
        "--project", type=str, default=None, help="W&B project name (single, shared); overrides WANDB_PROJECT from .env"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Run name, e.g. nano_base / nano_augmented / small_base / small_augmented"
    )
    parser.add_argument("--epochs", type=int, default=250, help="Keep identical across all 4 runs")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Ultralytics/YOLO standard. Keep identical across all 4 runs."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Reasonable starting point for Nano/Small at 640px on 16GB VRAM, not yet verified on this dataset. "
        "Keep identical across all 4 runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Ultralytics default, matches most training setups. Try 0 if you hit BrokenPipeError/EOFError.",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    return parser.parse_args()


def log_person_metrics(trainer):
    """Log 'person'-prefixed aliases of the current metrics to W&B, explicit for a single-class dashboard.

    nc=1 ("person"), so the mean-over-classes values below already ARE the per-class "person" values. Runs
    after Ultralytics' own on_fit_epoch_end (see README_EXPERIMENTS.md section 9), so this queues (commit=False)
    and lets that callback's commit=True flush both into the same step.
    """
    import wandb

    box = trainer.validator.metrics.box
    p, r, map50, map50_95 = box.mp, box.mr, box.map50, box.map
    f1 = 2 * p * r / (p + r + 1e-9)

    # ConfusionMatrix.matrix is (nc+1, nc+1); with nc=1, index 0 = "person", index 1 = "background".
    cm = trainer.validator.metrics.confusion_matrix.matrix
    tp, fp, fn = cm[0, 0], cm[0, 1], cm[1, 0]
    accuracy = tp / (tp + fp + fn + 1e-9)  # Jaccard index TP/(TP+FP+FN); no classification-style accuracy applies here

    wandb.run.log(
        {
            "person/precision": p,
            "person/recall": r,
            "person/f1_score": f1,
            "person/mAP50": map50,
            "person/mAP50-95": map50_95,
            "person/iou_at_0.5": map50,  # dataset-level detection rate at IoU>=0.5 == mAP50 for a single class
            "person/accuracy": accuracy,
        },
        step=trainer.epoch + 1,
        commit=False,
    )


def verify_gpu():
    """Print CUDA/GPU diagnostics and fail fast if the RTX 5060 Ti is not visible to PyTorch."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. Verify the NVIDIA driver is installed and that torch was "
            "installed from the CUDA wheel index (see requirements-windows.txt), not the CPU wheel."
        )
    print(f"torch.cuda.is_available()       = {torch.cuda.is_available()}")
    print(f"torch.version.cuda              = {torch.version.cuda}")
    print(f"torch.cuda.get_device_name(0)   = {torch.cuda.get_device_name(0)}")
    print(f"torch.cuda.get_device_capability(0) = {torch.cuda.get_device_capability(0)}")


def main():
    """Load credentials from .env, verify the GPU, and run one YOLO11 (Nano/Small) training on VisDrone."""
    args = parse_args()
    load_dotenv()  # reads .env from the current working directory (repo root)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY not set. Copy .env.example to .env at the repo root and fill it in.")

    project = args.project or os.environ.get("WANDB_PROJECT")
    if not project:
        raise RuntimeError("No W&B project set. Pass --project or set WANDB_PROJECT in .env.")

    import wandb

    # No explicit wandb.login() call: WANDB_API_KEY is already in os.environ from load_dotenv() above, and
    # wandb.init() picks it up on its own. This also sidesteps wandb.login()'s stricter key-format validation
    # in some versions (rejects keys that don't match the legacy personal-key length), which service-account
    # keys can trip even though they're valid — exporting the env var and letting init() authenticate is the
    # more robust path regardless of wandb version.
    #
    # Start the W&B run ourselves so every experiment lands in the SAME project (e.g. an existing "YOLOv11"
    # project), distinguished only by --name. If we didn't do this, Ultralytics' built-in wb.py callback
    # would call wandb.init(project=trainer.args.project, ...) itself using the local results `project`
    # folder as the W&B project name, creating a new project per run.
    wandb.init(project=project, name=args.name, config=vars(args))

    verify_gpu()

    from ultralytics import YOLO, settings

    settings.update({"wandb": True})  # enables ultralytics/utils/callbacks/wb.py (mAP, PR/F1 curves, losses, etc.)

    model = YOLO(args.model)
    model.add_callback("on_fit_epoch_end", log_person_metrics)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,  # lower to 0 if this triggers BrokenPipeError/EOFError from multiprocessing on Windows
        amp=True,
        plots=True,  # required for Ultralytics to log PR/F1 curves to W&B
        name=args.name,
        exist_ok=True,
        # No `project=` here: leaving it unset keeps local results under runs/detect/<name>,
        # independent from the W&B project set above via wandb.init().
        # No other hyperparameter is set here: optimizer, lr0, mosaic, mixup, fliplr, etc. all come
        # from ultralytics/cfg/default.yaml so all 4 runs (Nano/Small x base/augmented) stay comparable.
    )


if __name__ == "__main__":
    main()
