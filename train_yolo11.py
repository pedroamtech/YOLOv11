# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Train YOLO11l on VisDrone (single 'person' class) on Windows, tracked in Weights & Biases.

Usage (PowerShell, from the repo root):
    python train_yolo11.py `
        --data data\visdrone_base.yaml `
        --name base_run

    python train_yolo11.py `
        --data data\visdrone_augmented.yaml `
        --name augmented_run

Both commands above pass identical --epochs/--imgsz/--batch/--workers so the two experiments stay
comparable; only --data (and therefore --name) differs. Both runs report to the SAME W&B project
(--project / WANDB_PROJECT, e.g. an existing "YOLOv11" project) and are told apart by --name, not
by separate auto-created projects. See README_EXPERIMENTS.md.
"""

import argparse
import os

import torch
from dotenv import load_dotenv


def parse_args():
    """Parse command-line arguments for a single training run."""
    parser = argparse.ArgumentParser(description="Train YOLO11l on VisDrone (Windows, RTX 5060 Ti, 16GB VRAM)")
    parser.add_argument("--data", type=str, required=True, help="Path to visdrone_base.yaml or visdrone_augmented.yaml")
    parser.add_argument("--model", type=str, default="yolo11l.pt", help="Checkpoint to fine-tune from")
    parser.add_argument(
        "--project", type=str, default=None, help="W&B project name (single, shared); overrides WANDB_PROJECT from .env"
    )
    parser.add_argument("--name", type=str, required=True, help="Run name, e.g. base_run / augmented_run")
    parser.add_argument("--epochs", type=int, default=250, help="Keep identical across both experiments")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="720p source imagery; 1280 is the proven sweet spot for VisDrone-scale small objects "
        "(~9pp higher mAP50 than 640 in community benchmarks). Keep identical across experiments.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="-1 = AutoBatch (targets 60%% GPU memory), the standard baseline for VisDrone-scale training. Run "
        "Experiment 1 first, read the batch size AutoBatch prints, then pass that same fixed number as --batch "
        "to BOTH experiments so the comparison stays apples-to-apples.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Ultralytics default, matches most training setups. Try 0 if you hit BrokenPipeError/EOFError.",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device index")
    return parser.parse_args()


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
    """Load credentials from .env, verify the GPU, and run one YOLO11l training on VisDrone."""
    args = parse_args()
    load_dotenv()  # reads .env from the current working directory (repo root)

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if not wandb_api_key:
        raise RuntimeError("WANDB_API_KEY not set. Copy .env.example to .env at the repo root and fill it in.")

    project = args.project or os.environ.get("WANDB_PROJECT")
    if not project:
        raise RuntimeError("No W&B project set. Pass --project or set WANDB_PROJECT in .env.")

    import wandb

    wandb.login(key=wandb_api_key)
    # Start the W&B run ourselves so every experiment lands in the SAME project (e.g. an existing
    # "YOLOv11" project), distinguished only by --name. If we didn't do this, Ultralytics' built-in
    # wb.py callback would call wandb.init(project=trainer.args.project, ...) itself using the local
    # results `project` folder as the W&B project name, creating a new project per run.
    wandb.init(project=project, name=args.name, config=vars(args))

    verify_gpu()

    from ultralytics import YOLO, settings

    settings.update({"wandb": True})  # enables ultralytics/utils/callbacks/wb.py (mAP, PR/F1 curves, losses, etc.)

    model = YOLO(args.model)
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
        # from ultralytics/cfg/default.yaml so Experiment 1 and Experiment 2 remain comparable.
    )


if __name__ == "__main__":
    main()
