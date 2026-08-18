# VisDrone (person) — Experimentos de entrenamiento YOLO11m en Windows

Configuración local de entrenamiento en Windows para `yolo11m.pt` sobre VisDrone reducido a una
sola clase ("person"), comparando un dataset base contra una copia con aumentado offline, con
seguimiento en Weights & Biases. Estos archivos viven en la raíz del repo (mismo esquema que el
repo hermano [YOLOv12](https://github.com/pedroamtech/YOLOv12): `train_yolo12.py`,
`requirements-windows.txt`, `README_EXPERIMENTS.md`, `.env`/`.env.example`, `data/*.yaml`) y son
aditivos: no modifican nada bajo `ultralytics/` ni el `requirements.txt` original.

## Cómo usar esta guía

Sigue el orden de las secciones para evitar errores de entorno o de ejecución:

1. **Preparación del entorno** (§1-4): hardware requerido, diferencias entre `requirements.txt`
   y `requirements-windows.txt`, creación del entorno Anaconda, e instalación manual paso a paso.
2. **Configuración de parámetros** (§5-7): estructura de directorios y datasets esperada,
   hiperparámetros que se mantienen idénticos entre los dos experimentos, y credenciales de W&B.
3. **Ejecución** (§8): los dos comandos de entrenamiento, uno por dataset (base y aumentado).
4. **Verificación de resultados** (§9-10): qué métricas quedan registradas en W&B y cómo
   confirmar que la GPU quedó bien configurada antes de entrenar.

Si algo falla o los resultados difieren de lo documentado, revisa primero **§2**: ahí están los
dos problemas que ya se diagnosticaron y resolvieron en este mismo hardware (el fallo de `cu124`
con la RTX 5060 Ti, y el build de `stringzilla` en Windows) antes de asumir que es un problema
nuevo.

## 1. Hardware y entorno

| Componente | Especificación |
|---|---|
| Sistema operativo | Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM (Blackwell, compute capability sm_120) |
| CUDA Toolkit / driver | 13.3 |
| Framework | PyTorch con soporte CUDA |
| Modelo | YOLO11 Medium (`yolo11m.pt`), arquitectura C3k2/C2PSA — se eligió Medium en vez de Large por menor requerimiento de cómputo/VRAM |

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

  **Verificado en hardware real** (2026-08-17): se reprodujo el fallo exacto con `torch
  2.6.0+cu124` — `torch.cuda.is_available()` devolvía `True`, pero `torch.randn(4,4).cuda() @ ...`
  lanzaba `RuntimeError: CUDA error: no kernel image is available for execution on the device`,
  con la advertencia previa de PyTorch listando soporte hasta `sm_90` (no incluye `sm_120`).
  Tras reinstalar con el comando de la sección 4, `torch 2.11.0+cu128` reconoció la RTX 5060 Ti
  (`get_device_capability(0) = (12, 0)`) y ejecutó la misma operación en GPU sin error.

> **Problema conocido en Windows: build de `stringzilla` falla (`Microsoft Visual
> C++ 14.0 or greater is required`)**. `albumentations` **no** está pineado en
> `requirements-windows.txt` de este repo (Ultralytics lo trata como opcional
> y lo auto-instala en tiempo de ejecución si detecta que hace falta para
> ciertos aumentos); pero si lo instalas manualmente o el auto-install de
> Ultralytics lo dispara, arrastra `albucore`, que exige `stringzilla>=3.10.4`.
> Desde su serie 2.x, `stringzilla` **dejó de publicar wheels precompilados
> para Windows** en PyPI (los últimos `win_amd64` disponibles son de la serie
> 1.2.x, por debajo del mínimo que pide `albucore`), así que `pip` intenta
> compilarlo desde el código fuente y falla sin el compilador de Microsoft
> C++. Es el mismo problema documentado en el repo hermano
> [YOLOv12](https://github.com/pedroamtech/YOLOv12/blob/main/README_EXPERIMENTS.md#2-diferencias-requirementstxt-original-vs-requirements-windowstxt-nuevo) —
> no es específico de `yolo11l.pt` ni de `yolo12l.pt`, es una limitación del
> paquete `stringzilla` en Windows que afecta a cualquier modelo de
> Ultralytics que termine dependiendo de `albumentations`.
>
> **Fix (verificado)**: instalar *Build Tools for Visual Studio*
> (https://visualstudio.microsoft.com/visual-cpp-build-tools/) **no basta por
> sí solo** — el instalador base no incluye el compilador de C++. Abre
> **"Visual Studio Installer"**, elige **Modificar** sobre "Visual Studio
> Build Tools", y en la pestaña *Workloads* marca explícitamente **"Desktop
> development with C++"** (trae MSVC v143 + Windows SDK). Sin ese workload
> marcado, `cl.exe` no existe en el sistema y el error persiste aunque el
> instalador ya se haya "completado". Tras instalar el workload, cierra todas
> las ventanas de PowerShell abiertas, abre una nueva, reactiva el entorno
> conda (`conda activate yolov11`) y reintenta la instalación — `stringzilla`
> es código SIMD portable en C/C++ y compila sin problemas una vez presente
> el compilador.

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
`workers`, `amp`, `device` — ver §8. Los valores por defecto de `train_yolo11.py` son `imgsz=640`
(estándar YOLO/Ultralytics), `batch=16` y `workers=8` — ver el razonamiento en §8.

## 7. Credenciales W&B (seguras, sin hardcodear)

- `.env.example` (versionado en git, sin secretos reales) documenta las variables requeridas.
- `.env` (NO versionado, ver `.gitignore`) contiene tu `WANDB_API_KEY` real.
- `train_yolo11.py` carga `.env` con `python-dotenv` y llama a `wandb.login(key=...)` antes de
  entrenar; si `WANDB_API_KEY` falta, el script aborta con un mensaje claro en vez de entrenar
  sin tracking.
- Ambos experimentos reportan al **mismo proyecto W&B** (uno ya creado por ti, ej. `YOLOv11`) —
  se distinguen por `--name` (`base_run` / `augmented_run`), no creando un proyecto nuevo por
  corrida. `train_yolo11.py` llama a `wandb.init(project=..., name=...)` **antes** de
  `model.train()`, así que el callback nativo `wb.py` de Ultralytics detecta el run ya activo
  (`if not wb.run:`) y reutiliza ese mismo proyecto en vez de crear uno propio a partir de la
  carpeta local de resultados.

## 8. Ejecutar ambos entrenamientos (PowerShell)

Con el entorno `yolov11` (§3) activado:

```powershell
# Copia la plantilla de credenciales y completa tu WANDB_API_KEY real
Copy-Item .env.example .env
notepad .env

# Experimento 1 — dataset base
python train_yolo11.py `
    --data data\visdrone_base.yaml `
    --name base_run `
    --epochs 250 --imgsz 640 --batch 16 --workers 8

# Experimento 2 — dataset con aumentado offline (mismos hiperparámetros, distinto --data/--name)
python train_yolo11.py `
    --data data\visdrone_augmented.yaml `
    --name augmented_run `
    --epochs 250 --imgsz 640 --batch 16 --workers 8
```

Ambas corridas usan `WANDB_PROJECT=YOLOv11` desde `.env` (o pasa `--project <tu-proyecto>` para
sobreescribirlo). Resultados locales guardados en carpetas independientes: `runs\detect\base_run\`
y `runs\detect\augmented_run\` (ya cubiertas por la regla `runs/` del `.gitignore`) — separados de
a qué proyecto de W&B reportan.

> **Por qué estos valores (`imgsz=640`, `batch=16`, `workers=8`)**: `imgsz=640` es el estándar
> YOLO/Ultralytics (`ultralytics/cfg/default.yaml`). `batch=16` es un punto de partida razonable
> — a 640px hay mucho más margen de VRAM que a 1280px (donde `batch=8` ya daba OOM en pruebas
> anteriores); no está verificado aún específicamente en este dataset, así que si aparece `OOM`
> baja `--batch`, y si sobra VRAM puedes subirlo — solo usa el **mismo** valor en ambos
> experimentos para que la comparación siga siendo válida. `workers=8` no cambia: ya estaba bien
> dimensionado para tu CPU de 24 hilos, independientemente de la resolución; solo baja a `0` si
> aparece `BrokenPipeError`/`EOFError` por multiprocessing en Windows.

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
