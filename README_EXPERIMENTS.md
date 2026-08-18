# VisDrone (person) — Experimentos de entrenamiento YOLO11l en Windows

Configuración local de entrenamiento en Windows para `yolo11l.pt` sobre VisDrone reducido a una
sola clase ("person"), comparando un dataset base contra una copia con aumentado offline, con
seguimiento en Weights & Biases. Estos archivos viven en la raíz del repo (mismo esquema que el
repo hermano [YOLOv12](https://github.com/pedroamtech/YOLOv12): `train_yolo12.py`,
`requirements-windows.txt`, `README_EXPERIMENTS.md`, `.env`/`.env.example`, `data/*.yaml`) y son
aditivos: no modifican nada bajo `ultralytics/` ni el `requirements.txt` original.

## 1. Hardware y entorno

| Componente | Especificación |
|---|---|
| Sistema operativo | Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM (Blackwell, compute capability sm_120) |
| CUDA Toolkit / driver | 13.3 |
| Framework | PyTorch con soporte CUDA |
| Modelo | YOLO11 Large (`yolo11l.pt`), arquitectura C3k2/C2PSA |

## 2. Diferencias: `requirements.txt` (original) vs `requirements-windows.txt` (nuevo)

Este repo no versiona un `requirements.txt` en la raíz (está en `.gitignore`; las dependencias
viven en `pyproject.toml`). `requirements-windows.txt` es una lista **independiente, de
instalación manual**, solo para estos experimentos:

- **PyTorch se instala aparte, primero, desde el índice de wheels CUDA** — ver la corrección de
  abajo.
- **No había paquetes exclusivos de Linux que quitar.** Las dependencias base de este
  `pyproject.toml` no incluyen `triton` ni `flash-attn` (a diferencia del repo hermano YOLOv12,
  donde sí aparecían como extras y se retiraron). `requirements-windows.txt` es entonces el set
  de runtime seguro para Windows más dos añadidos: `wandb` y `python-dotenv`, para el tracking
  con credenciales.
- **Corrección sobre el índice `cu124` originalmente solicitado:** los wheels `cu124` de PyTorch
  son anteriores al soporte de Blackwell (RTX serie 50) y fallan en la RTX 5060 Ti con *"CUDA
  error: no kernel image is available for execution on the device"* aunque
  `torch.cuda.is_available()` devuelva `True`. El driver CUDA 13.3 es retrocompatible con
  runtimes CUDA de PyTorch más antiguos, así que la solución no es un índice `cu13x` (no hace
  falta que exista), sino un wheel que sí incluya kernels `sm_120`: **`cu128`** (PyTorch ≥2.7).
  Comando exacto en la sección 4.

## 3. Creación del entorno virtual (Anaconda)

```powershell
conda create -n yolov11 python=3.11 -y
conda activate yolov11
python -m pip install --upgrade pip
```

`python=3.11` coincide con el piso de soporte activo de este repo y tiene buena cobertura de
wheels precompilados para `torch`/`onnxruntime`. Verifica que el intérprete correcto esté activo:

```powershell
where.exe python   # debe apuntar dentro de \Anaconda3\envs\yolov11\ o \Miniconda3\envs\yolov11\
```

Para desmontarlo después: `conda deactivate` y luego `conda env remove -n yolov11`.

## 4. Instalación manual en Windows (ningún script automático)

Con el entorno `yolov11` activado (sección 3):

```powershell
# 1) PyTorch con soporte CUDA para la RTX 5060 Ti — instalar ANTES del paso 2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Si el build estable aún reporta "no kernel image is available for execution on the device"
# en tu driver específico, usa el build nightly cu128 como alternativa:
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 2) Resto de dependencias para estos experimentos
pip install -r requirements-windows.txt

# 3) Tu propio clon del repo, editable (usa ultralytics/ de este checkout, no de PyPI)
pip install -e . --no-deps
```

`--no-deps` en el paso 3 evita que pip vuelva a resolver `torch`/`torchvision` contra el índice
por defecto (no-CUDA) de PyPI después de que el paso 1 ya instaló el build CUDA correcto.

## 5. Estructura de directorios esperada

```
GitHub/
├── YOLOv11/                            ← este repo
│   ├── requirements-windows.txt
│   ├── train_yolo11.py
│   ├── README_EXPERIMENTS.md
│   ├── .env / .env.example
│   ├── data/
│   │   ├── visdrone_base.yaml
│   │   └── visdrone_augmented.yaml
│   └── runs/                           ← resultados (en .gitignore)
└── datasets/                           ← en .gitignore, la puebla el usuario
    ├── VisDrone_person_base/
    │   ├── images/{train,val,test}/*.jpg
    │   └── labels/{train,val,test}/*.txt   # índice de clase siempre 0 ("person")
    └── VisDrone_person_augmented/
        ├── images/{train,val,test}/*.jpg   # tu copia con aumentado offline
        └── labels/{train,val,test}/*.txt
```

`data/visdrone_base.yaml` y `data/visdrone_augmented.yaml` usan un `path:` **sin prefijo** (sin
`../`), igual que la convención propia de Ultralytics (ver
`ultralytics/cfg/datasets/VisDrone.yaml`): una ruta relativa se resuelve contra el ajuste global
`datasets_dir`, que por defecto es `<carpeta padre del repo>/datasets` — de ahí el layout de
arriba. Verifícalo con:

```powershell
python -c "from ultralytics import settings; print(settings['datasets_dir'])"
```

Si tu copia local vive en otro lado, muévela bajo `datasets_dir`, cambia `datasets_dir` con
`yolo settings datasets_dir=...`, o pon una ruta absoluta en `path:`. Ambos YAML son idénticos
salvo `path:` — `nc: 1`, `names: [person]` y todos los hiperparámetros son iguales entre los dos
experimentos.

## 6. Hiperparámetros (idénticos en ambos experimentos)

`train_yolo11.py` **no** pasa overrides de `lr0`, `optimizer`, `mosaic`, `mixup`, `fliplr`,
`hsv_*`, `degrees`, etc. — todos se heredan sin modificar de `ultralytics/cfg/default.yaml`,
entre ellos:

| Parámetro | Valor por defecto |
|---|---|
| `optimizer` | `auto` |
| `lr0` / `lrf` | `0.01` / `0.01` |
| `momentum` | `0.937` |
| `weight_decay` | `0.0005` |
| `mosaic` | `1.0` |
| `mixup` | `0.0` |
| `fliplr` | `0.5` |
| `hsv_h/s/v` | `0.015 / 0.7 / 0.4` |
| `close_mosaic` | `10` (últimas 10 épocas sin mosaic) |
| `patience` | `100` |

Solo se controlan parámetros de **ejecución/hardware** (no de red): `epochs`, `imgsz`, `batch`,
`workers`, `amp`, `device` — ver §8.

## 7. Credenciales W&B (seguras, sin hardcodear)

- `.env.example` (versionado en git, sin secretos reales) documenta las variables requeridas.
- `.env` (NO versionado, ver `.gitignore`) contiene tu `WANDB_API_KEY` real.
- `train_yolo11.py` carga `.env` con `python-dotenv` y llama a `wandb.login(key=...)` antes de
  entrenar; si `WANDB_API_KEY` falta, el script aborta con un mensaje claro en vez de entrenar
  sin tracking.
- Cada experimento reporta a su **propio proyecto W&B**, indicado con `--project` en cada corrida
  (ver §8) — Base y Augmented nunca se mezclan en el mismo proyecto.

## 8. Ejecutar ambos entrenamientos (PowerShell)

Con el entorno `yolov11` (§3) activado:

```powershell
# Copia la plantilla de credenciales y completa tu WANDB_API_KEY real
Copy-Item .env.example .env
notepad .env

# Experimento 1 — dataset base
python train_yolo11.py `
    --data data\visdrone_base.yaml `
    --project VisDrone-YOLO11L-Base `
    --name base_run `
    --epochs 250 --imgsz 1280 --batch 8 --workers 2

# Experimento 2 — dataset con aumentado offline (mismos hiperparámetros, distinto --data/--project)
python train_yolo11.py `
    --data data\visdrone_augmented.yaml `
    --project VisDrone-YOLO11L-Augmented `
    --name augmented_run `
    --epochs 250 --imgsz 1280 --batch 8 --workers 2
```

Resultados guardados en carpetas independientes: `runs\detect\base_run\` y
`runs\detect\augmented_run\` (ya cubiertas por la regla `runs/` del `.gitignore`).

> **Resolución de imagen (720p)**: las imágenes fuente son 1280×720. En modo `train`, Ultralytics
> recibe `imgsz` como un único entero que define el lado largo del letterbox cuadrado (aquí
> `1280`); el lado corto se rellena (padding) en vez de recortarse o deformarse, así que no se
> pierde detalle. `batch=8` a `imgsz=1280` es un punto de partida conservador para `yolo11l.pt` en
> 16 GB de VRAM (subir la resolución de 640 a 1280 cuesta bastante más memoria que el batch en sí);
> si aparece `OOM` baja `--batch` a `4`, y si sobra VRAM puedes subirlo — solo usa el **mismo**
> valor en ambos experimentos para que la comparación siga siendo válida.

Si aparece `BrokenPipeError` / `EOFError` (multiprocessing en Windows), baja `--workers` a `0`.

## 9. Métricas registradas en W&B

La integración nativa de Ultralytics (`ultralytics/utils/callbacks/wb.py`), activada vía
`settings.update({"wandb": True})` en `train_yolo11.py`, ya registra automáticamente por época
(sin necesidad de llamadas manuales a `wandb.log()`):

- `metrics/precision(B)`, `metrics/recall(B)`
- `metrics/mAP50(B)`, `metrics/mAP50-95(B)`
- Pérdidas de entrenamiento y validación: `box_loss`, `cls_loss`, `dfl_loss`
- Curva Precision-Recall y curva F1-Confidence para la clase `person` (vía `_plot_curve`, tomadas
  de `trainer.validator.metrics.curves_results` al final del entrenamiento)
- Artefacto del mejor checkpoint (`best.pt`)

Dos métricas del pedido original necesitan una aclaración, porque detección de objetos no las
produce de forma nativa como un clasificador:

- **IoU**: no hay un escalar único de "IoU" por época — IoU es lo que mAP mide *sobre*: mAP@0.5 es
  mAP con umbral de IoU 0.5, mAP@0.5:0.95 es el promedio entre IoU 0.5 y 0.95. Por eso ambas ya se
  registran arriba en vez de un número de IoU aparte.
- **Accuracy**: no es una métrica estándar de detección de objetos (no hay un "total" fijo sobre
  el cual dividir predicciones correctas, a diferencia de clasificación). Precision, Recall y F1
  (derivable de la curva PR ya registrada) son los sustitutos estándar y ya se registran.

## 10. Verificación de GPU (incluida en el script)

Antes de cada entrenamiento, `train_yolo11.py` imprime y valida:

```python
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
```

Si `torch.cuda.is_available()` es `False`, el script aborta con `RuntimeError` antes de cargar
credenciales de W&B o el dataset.
