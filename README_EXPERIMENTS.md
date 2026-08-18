# VisDrone (person) — YOLO11l Windows Training Experiments

Local Windows training setup for `yolo11l.pt` on a single-class ("person") VisDrone dataset,
comparing a base dataset against an offline-augmented copy, tracked in Weights & Biases. These
files live at the repo root (mirroring the layout used in the sibling
[YOLOv12](https://github.com/pedroamtech/YOLOv12) repo: `train_yolo12.py`,
`requirements-windows.txt`, `README_EXPERIMENTS.md`, `.env.example`, `data/*.yaml`) and are
additive: nothing under `ultralytics/` is modified, and `requirements.txt` is untouched.

## 1. Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM (Blackwell, compute capability sm_120) |
| CUDA Toolkit / driver | 13.3 |
| OS | Windows 11 |

## 2. `requirements-windows.txt` vs. the repo's `requirements.txt`

The repo does not track a root `requirements.txt` (it's git-ignored; dependencies live in
`pyproject.toml`). `requirements-windows.txt` is a **separate, manually-installed** list for this
experiment only. Differences from a plain `pip install -e .`:

- **PyTorch is installed separately, first, from a CUDA wheel index** — see the correction below.
- **No Linux-only packages were present to strip.** This repo's `pyproject.toml` base
  dependencies contain no `triton` or `flash-attn`; those only ever appear (in other projects,
  e.g. the sibling YOLOv12 repo) as optional/export extras. `requirements-windows.txt` is
  therefore just the Windows-safe runtime set plus two additions: `wandb` and `python-dotenv`,
  for credential-based experiment tracking.
- **Correction on the requested `cu124` index:** cu124 PyTorch wheels predate Blackwell (RTX
  50-series) kernel support. Installing torch from `cu124` on an RTX 5060 Ti passes
  `torch.cuda.is_available()` but fails at the first kernel launch with *"CUDA error: no kernel
  image is available for execution on the device."* The NVIDIA driver's CUDA 13.3 capability is
  backward-compatible with older PyTorch CUDA runtimes, so the fix is not a `cu13x` wheel index
  (doesn't need to exist) — it's using a wheel that ships sm_120 kernels, i.e. **`cu128`** (PyTorch
  ≥2.7). Install command in section 3.

## 3. Manual install (PowerShell)

### Step 1 — Anaconda / Miniconda environment

Same layout as the sibling [YOLOv12](https://github.com/pedroamtech/YOLOv12) repo's env setup:

```powershell
conda create -n yolov11 python=3.11 -y
conda activate yolov11
python -m pip install --upgrade pip
```

`python=3.11` matches this repo's actively-supported floor and has solid prebuilt-wheel coverage
for `torch`/`onnxruntime`. Verify the right interpreter is active before continuing:

```powershell
where.exe python   # should point inside \Anaconda3\envs\yolov11\ or \Miniconda3\envs\yolov11\
```

To tear it down later: `conda deactivate` then `conda env remove -n yolov11`.

### Steps 2–4

```powershell
# 2) PyTorch with CUDA support for the RTX 5060 Ti — install this BEFORE step 3.
#    Still via pip inside the conda env: conda-forge's pytorch build lags on new CUDA/arch
#    support (Blackwell/sm_120), so pip + the official CUDA wheel index is the reliable path.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# If this stable build still reports "no kernel image is available for execution on the device"
# on your specific RTX 5060 Ti driver, fall back to the nightly cu128 build:
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Everything else for these experiments
pip install -r requirements-windows.txt

# 4) Your cloned repo itself, editable (uses ultralytics/ from this checkout, not PyPI)
pip install -e . --no-deps
```

`--no-deps` in step 4 avoids pip trying to re-resolve `torch`/`torchvision` against PyPI's default
(non-CUDA) index after step 2 already installed the correct CUDA build.

## 4. Dataset structure

VisDrone, re-labeled to a single class, in standard Ultralytics YOLO format (normalized
`x_center y_center width height` in `[0, 1]`, one `.txt` per image):

```
nc: 1
names: ['person']
```

`data/visdrone_base.yaml` and `data/visdrone_augmented.yaml` use a **bare** `path:` (no `../`),
matching Ultralytics' own dataset convention (see `ultralytics/cfg/datasets/VisDrone.yaml`).
Ultralytics resolves a relative `path` against the global `datasets_dir` setting — check it with:

```powershell
python -c "from ultralytics import settings; print(settings['datasets_dir'])"
```

By default `datasets_dir` is `<parent of this repo checkout>/datasets`, so the two copies are
expected as siblings of the repo:

```
datasets/
├── VisDrone_person_base/
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}      # class index is always 0 ("person")
└── VisDrone_person_augmented/
    ├── images/{train,val,test}      # your offline-augmented copy of the same data
    └── labels/{train,val,test}
```

If your local copy lives elsewhere, either move it under `datasets_dir`, change `datasets_dir` via
`yolo settings datasets_dir=...`, or set `path:` in the yaml to an absolute path.

`data/visdrone_base.yaml` and `data/visdrone_augmented.yaml` differ **only** in `path:` — `nc`,
`names`, and every training hyperparameter are identical between the two runs, per the requirement
below.

## 5. Hyperparameters held constant across both experiments

`train_yolo11.py` passes only execution/hardware settings explicitly (`data`, `model`, `epochs`,
`imgsz`, `batch`, `device`, `workers`, `amp`, `plots`, `project`, `name`). It does **not** set
`lr0`, `optimizer`, `mosaic`, `mixup`, `fliplr`, or any other augmentation/optimization parameter —
those come from `ultralytics/cfg/default.yaml` and are therefore byte-for-byte identical between
Experiment 1 and Experiment 2, as long as you pass the same
`--epochs`/`--imgsz`/`--batch`/`--workers` to both commands (the example commands in section 7
already do this).

- `workers=2` (or `0`) avoids `BrokenPipeError`/`EOFError` from Windows multiprocessing.
- `amp=True` for mixed precision.
- `batch=16` at `imgsz=640` is a reasonable starting point for `yolo11l.pt` on 16 GB VRAM; if you
  hit an out-of-memory error, lower it (e.g. `--batch 8`) — just use the **same** value for both
  experiments so the comparison stays valid.

## 6. Metrics logged to Weights & Biases

Tracking uses Ultralytics' **built-in** W&B integration (`ultralytics/utils/callbacks/wb.py`),
enabled via `settings.update({"wandb": True})` in `train_yolo11.py` — no manual `wandb.log()`
calls are needed in the training loop. Each run automatically logs, per epoch and at the end of
training:

- **mAP@0.5** and **mAP@0.5:0.95** (`metrics/mAP50(B)`, `metrics/mAP50-95(B)`)
- **Precision** and **Recall** (`metrics/precision(B)`, `metrics/recall(B)`)
- **Box / classification / DFL loss** curves (train and val)
- **Precision-Recall curve** and **F1-confidence curve** for the `person` class (via
  `_plot_curve`, logged from `trainer.validator.metrics.curves_results` at `on_train_end`)
- Final best-weights model artifact

Two metrics named in the original request need a note, since detection models don't natively
produce them the way classifiers do:

- **IoU**: there's no single scalar "IoU" logged by the trainer; IoU is what mAP is computed
  *over* — mAP@0.5 = mAP at IoU threshold 0.5, mAP@0.5:0.95 = averaged over IoU 0.5–0.95. That's
  why both are already tracked above rather than a separate IoU number.
- **Accuracy**: not a standard object-detection metric (there's no fixed "total" to divide correct
  predictions by, unlike classification). Precision, Recall, and F1 (derivable from precision/recall
  per point on the logged PR curve) are the standard substitutes and are already tracked.

Each experiment reports to its **own W&B project** (see the `--project` flags in section 7), so
Base and Augmented runs never mix in the same project.

## 7. Running both experiments (PowerShell)

```powershell
# Copy the credentials template and fill in your real WANDB_API_KEY
Copy-Item .env.example .env
notepad .env

# Experiment 1 — base dataset
python train_yolo11.py `
    --data data\visdrone_base.yaml `
    --project VisDrone-YOLO11L-Base `
    --name base_run `
    --epochs 100 --imgsz 640 --batch 16 --workers 2

# Experiment 2 — offline-augmented dataset (identical hyperparameters, different --data/--project)
python train_yolo11.py `
    --data data\visdrone_augmented.yaml `
    --project VisDrone-YOLO11L-Augmented `
    --name augmented_run `
    --epochs 100 --imgsz 640 --batch 16 --workers 2
```

Results save to `runs\detect\base_run\` and `runs\detect\augmented_run\` respectively (both are
already git-ignored via the repo's `runs/` rule), and each run streams live to its own W&B
project.
