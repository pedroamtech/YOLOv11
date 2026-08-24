# YOLO11 Nano/Small sobre VisDrone (Windows) — Documentación de experimentos

Este documento describe el flujo de entrenamiento local en Windows que armé para `yolo11n.pt`/
`yolo11s.pt` sobre un dataset VisDrone de clase única (`person`), con tracking en Weights &
Biases. No modifica `requirements.txt` ni ningún archivo del paquete `ultralytics/` original.

> **Alcance actual: solo Nano y Small.** Empecé con `yolo11l.pt` (Large) — exigía demasiado
> cómputo/VRAM para este flujo — pasé a `yolo11m.pt` (Medium), y finalmente recorté el alcance del
> proyecto a Nano y Small únicamente (ver sección 9 para los números concretos de parámetros/
> FLOPs detrás de esa decisión). `--model yolo11m.pt` (o `yolo11l.pt`) sigue siendo técnicamente
> válido para retomarlos: el script no cambió, solo el plan de experimentos documentado acá (ver
> "Otros tamaños de modelo" al final de la sección 9).

## Instrucciones de ejecución

Sigue este orden para reproducir el experimento sin errores:

1. **Preparación del entorno**: crea el entorno conda (sección 3) e instala las dependencias —
   **no** el `requirements.txt` original del repo, sino `requirements-windows.txt` (sección 4),
   el archivo independiente armado para este flujo en Windows.
2. **Configuración de parámetros**: apunta `data/visdrone_base.yaml` y
   `data/visdrone_augmented.yaml` (sección 5) a tu dataset real, y copia `.env.example` a `.env`
   (sección 8) con tu `WANDB_API_KEY`/`WANDB_PROJECT` reales. No toques los hiperparámetros de red
   (sección 6) ni la metodología de fine-tuning (sección 7) — quedan idénticos en las cuatro
   corridas.
3. **Ejecución**: corre `train_yolo11.py` con los comandos de PowerShell de la sección 9 — cuatro
   corridas: Nano y Small, cada uno con Base y Augmented.
4. **Resolución de problemas**: si algo falla o los resultados difieren de lo esperado, revisa la
   sección 12 al final de este documento — reúne los problemas que encontré y resolví durante
   estas pruebas (fallo de CUDA en la RTX 5060 Ti, build de `stringzilla`, proyecto de W&B mal
   nombrado, entrenamiento lento por OOM silencioso).

## 1. Hardware y entorno

| Componente | Valor |
|---|---|
| SO | Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Ti — 16 GB VRAM |
| CUDA Toolkit | 13.3 (driver del sistema) |
| Framework | PyTorch + CUDA (wheel, ver sección 2) |
| Modelos | YOLO11 Nano (`yolo11n.pt`) y Small (`yolo11s.pt`) |

> **Nota sobre CUDA 13.3 y wheels de PyTorch (confirmado en esta máquina)**: los wheels oficiales
> de PyTorch se distribuyen con etiquetas `cuXXX` (p. ej. `cu124`, `cu128`) que empaquetan su
> propio runtime CUDA; no necesitan coincidir exactamente con la versión del CUDA Toolkit del
> sistema, solo requieren un **driver NVIDIA igual o más nuevo** que el mínimo exigido por ese
> runtime. La RTX 5060 Ti es arquitectura **Blackwell (compute capability `sm_120`)**. **`cu124`
> NO sirve**: lo probé (`torch==2.6.0+cu124`) y, aunque `torch.cuda.is_available()` devuelve
> `True` (por eso es engañoso — solo verifica que hay GPU + driver, no que el build tenga kernels
> para esa arquitectura), PyTorch advierte explícitamente `NVIDIA GeForce RTX 5060 Ti with CUDA
> capability sm_120 is not compatible with the current PyTorch installation` (soporta hasta
> `sm_90`, RTX 40) y cualquier operación real en GPU falla con `RuntimeError: CUDA error: no
> kernel image is available for execution on the device`. El fix confirmado es reinstalar con
> **`cu128`** (comando exacto en sección 4) — verifiqué que `torch 2.11.0+cu128` sí reconoce
> `sm_120` y ejecuta inferencia real en GPU sin error.

## 2. Diferencias: `requirements.txt` (original) vs `requirements-windows.txt` (nuevo)

Este repo no versiona un `requirements.txt` en la raíz (está en `.gitignore`; las dependencias
viven en `pyproject.toml`). `requirements-windows.txt` es una lista **independiente, de
instalación manual**, solo para estos experimentos:

| Paquete | Situación en `pyproject.toml` | `requirements-windows.txt` | Motivo |
|---|---|---|---|
| `torch` / `torchvision` | Dependencia base, sin índice CUDA fijo | **Excluidos del archivo** — se instalan aparte | Necesitan un build reciente con soporte `sm_120` (Blackwell) desde el índice CUDA correcto; instalarlos desde PyPI da un build CPU o sin soporte Blackwell |
| `triton`, `flash-attn` | No están en las dependencias base de este repo | Nada que quitar | A diferencia del repo hermano YOLOv12 (donde sí aparecían como extras y los retiré ahí), este `pyproject.toml` no los lista para empezar |
| `python-dotenv` | No está | **Añadido** | Carga segura de `WANDB_API_KEY`/`WANDB_PROJECT` desde `.env` |
| `wandb` | No está | **Añadido** | Tracking de métricas |
| Resto (`numpy`, `opencv-python`, `matplotlib`, `pyyaml`, etc.) | Igual | Igual | Sin cambios, multiplataforma |

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
> C++ instalado. No es un problema de este repo ni de
> `requirements-windows.txt`: es una limitación actual del paquete
> `stringzilla` en Windows, y **afecta a cualquier versión de modelo de
> Ultralytics** (YOLOv12, YOLO11 Nano/Small/Medium/Large, etc.) que use
> `albumentations` en Windows — no es específico de ningún tamaño de modelo
> en particular. Es el mismo problema que documenté en el repo hermano
> [YOLOv12](https://github.com/pedroamtech/YOLOv12/blob/main/README_EXPERIMENTS.md#2-diferencias-requirementstxt-original-vs-requirements-windowstxt-nuevo).
>
> **Fix (verificado)**: instalar *Build Tools for Visual Studio*
> (https://visualstudio.microsoft.com/visual-cpp-build-tools/) **no basta
> por sí solo** — el instalador base no incluye el compilador de C++. Abre
> **"Visual Studio Installer"**, elige **Modificar** sobre "Visual Studio
> Build Tools", y en la pestaña *Workloads* marca explícitamente
> **"Desktop development with C++"** (trae MSVC v143 + Windows SDK). Sin ese
> workload marcado, `cl.exe` no existe en el sistema y el error persiste
> aunque el instalador ya se haya "completado". Después de instalar el
> workload, cierra todas las ventanas de PowerShell abiertas (para refrescar
> el entorno), abre una nueva, activa el entorno conda (`conda activate
> yolov11`) y reintenta `pip install -r requirements-windows.txt` —
> `stringzilla` es código SIMD portable en C/C++ y compila sin problemas una
> vez presente el compilador.

## 3. Creación del entorno virtual (Anaconda)

Todo este flujo (instalación de dependencias, entrenamiento, tracking) lo armo dentro de un
entorno conda dedicado, para no interferir con otras instalaciones de Python/PyTorch en el
sistema.

```powershell
# 1) Crear el entorno con Python 3.11 (coincide con la versión objetivo del repo; onnxruntime-gpu,
#    torch y el resto de wheels tienen soporte sólido)
conda create -n yolov11 python=3.11 -y

# 2) Activar el entorno (repetir esto en cada sesión de PowerShell nueva antes de instalar
#    dependencias o lanzar train_yolo11.py)
conda activate yolov11

# 3) Confirmar que el entorno activo es el correcto
python --version
where.exe python
```

> `where.exe python` debe apuntar a una ruta dentro de `...\anaconda3\envs\yolov11\python.exe` (o
> `...\miniconda3\envs\...`). Si apunta al Python global del sistema, el entorno no está activado.

Para desactivar el entorno al terminar la sesión: `conda deactivate`. Para eliminarlo por
completo (reinstalación desde cero): `conda env remove -n yolov11`.

## 4. Instalación manual en Windows (ningún script automático)

Con el entorno `yolov11` de la sección 3 activado, ejecuta en PowerShell:

```powershell
# 1) Actualizar pip
python -m pip install --upgrade pip

# 2) Instalar PyTorch con soporte CUDA — cu128, CONFIRMADO para RTX 5060 Ti / Blackwell
#    (cu124 se probó y NO funciona: PyTorch reporta sm_120 como no soportado)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2b) SOLO SI cu128 estable también falla en tu driver específico (variante nightly, no debería
#     hacer falta):
# pip uninstall -y torch torchvision
# pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Instalar el resto de dependencias (archivo independiente, sin tocar requirements.txt)
pip install -r requirements-windows.txt

# 4) Instalar este repo (YOLOv11) en modo editable, sin re-resolver torch/torchvision
pip install -e . --no-deps
```

> Uso `--no-deps` en el paso 4 para evitar que pip vuelva a resolver `torch`/`torchvision` contra
> el índice por defecto (no-CUDA) de PyPI después de que el paso 2 ya instaló el build CUDA
> correcto.

## 5. Estructura de directorios esperada

`data/visdrone_base.yaml` y `data/visdrone_augmented.yaml` usan un `path:` **sin prefijo** (sin
`../`), igual que la convención propia de Ultralytics (ver
`ultralytics/cfg/datasets/VisDrone.yaml`): una ruta relativa se resuelve contra el ajuste global
`datasets_dir`, que por defecto es `<carpeta padre del repo clonado>/datasets`. Verifícalo con:

```powershell
python -c "from ultralytics import settings; print(settings['datasets_dir'])"
```

```
GitHub/
├── YOLOv11/                            ← este repo clonado
│   ├── requirements.txt                (original, sin tocar)
│   ├── requirements-windows.txt        (nuevo)
│   ├── train_yolo11.py                 (nuevo)
│   ├── README_EXPERIMENTS.md           (este archivo)
│   ├── .env / .env.example
│   ├── data/
│   │   ├── visdrone_base.yaml
│   │   └── visdrone_augmented.yaml
│   └── runs/                           ← resultados (gitignored)
│       └── detect/
│           ├── nano_base/
│           ├── nano_augmented/
│           ├── small_base/
│           └── small_augmented/
└── datasets/                           ← gitignored, la puebla el usuario
    ├── VisDrone_person_base/
    │   ├── images/{train,val,test}/*.jpg
    │   └── labels/{train,val,test}/*.txt   # índice de clase siempre 0 ("person")
    └── VisDrone_person_augmented/
        ├── images/{train,val,test}/*.jpg   # tu copia con aumentado offline
        └── labels/{train,val,test}/*.txt
```

Si tu copia real del dataset vive en otro lado (por ejemplo, una ruta absoluta fuera de esta
convención), tienes tres opciones: muévela bajo `datasets_dir`, cambia `datasets_dir` con
`yolo settings datasets_dir=...`, o pon una ruta absoluta directamente en `path:` dentro del
`.yaml` — Ultralytics usa una ruta absoluta tal cual, sin pasar por `datasets_dir`.

- **Clases**: `nc: 1`, `names: ['person']` (índice `0`) en ambos `.yaml`.
- **Dataset base** (`visdrone_base.yaml`) apunta al dataset sin preprocesamiento adicional.
- **Dataset aumentado** (`visdrone_augmented.yaml`) apunta a una copia de las mismas imágenes,
  pasadas por un pipeline de aumento de datos *offline*, previo al entrenamiento. Los aumentos
  *on-the-fly* de YOLO (mosaic, mixup, fliplr, hsv, etc. — sección 6) se aplican igual en ambos
  casos, con los mismos valores por defecto de `ultralytics/cfg/default.yaml` en las cuatro
  corridas — la única variable entre dataset base y aumentado es el contenido físico de
  imágenes/etiquetas.

## 6. Hiperparámetros (idénticos en las cuatro corridas)

`train_yolo11.py` **no** pasa overrides de ningún hiperparámetro de red — todos se heredan sin
modificar de `ultralytics/cfg/default.yaml`. Agrupados por categoría:

### 6.1 Optimización / entrenamiento

| Parámetro | Valor por defecto |
|---|---|
| `optimizer` | `auto` |
| `lr0` / `lrf` | `0.01` / `0.01` |
| `momentum` | `0.937` |
| `weight_decay` | `0.0005` |
| `warmup_epochs` | `3.0` |
| `warmup_momentum` | `0.8` |
| `warmup_bias_lr` | `0.1` |
| `cos_lr` | `False` (decay lineal, no coseno) |

> Estos son los valores tal cual están en `ultralytics/cfg/default.yaml` — con `optimizer: auto`,
> varios de ellos se recalculan o se sobreescriben en tiempo de ejecución (`optimizer`, `lr0`,
> `momentum` efectivo, `warmup_bias_lr`). El detalle completo, con líneas de código exactas, está
> en la sección 7.

### 6.2 Función de pérdida

| Parámetro | Valor por defecto |
|---|---|
| `box` | `7.5` (peso de la pérdida de caja) |
| `cls` | `0.5` (peso de la pérdida de clase) |
| `dfl` | `1.5` (peso de distribution focal loss) |

### 6.3 Aumento de datos clásico (on-the-fly, activo en detección)

| Parámetro | Valor por defecto |
|---|---|
| `hsv_h` / `hsv_s` / `hsv_v` | `0.015` / `0.7` / `0.4` |
| `degrees` | `0.0` (rotación) |
| `translate` | `0.1` |
| `scale` | `0.5` |
| `shear` | `0.0` |
| `perspective` | `0.0` |
| `flipud` | `0.0` (flip vertical) |
| `fliplr` | `0.5` (flip horizontal) |
| `bgr` | `0.0` |
| `mosaic` | `1.0` |
| `mixup` | `0.0` |
| `copy_paste` | `0.0` |
| `copy_paste_mode` | `flip` |
| `close_mosaic` | `10` (últimas 10 épocas sin mosaic) |

### 6.4 Aumento adicional vía Albumentations (fijo en código, no en `default.yaml`)

Además de lo anterior, esta versión de Ultralytics aplica siempre esta transformación
`albumentations` — visible en el log de cada corrida — con probabilidades **hardcodeadas** en
`ultralytics/data/augment.py:2100-2109`, no configurables vía `default.yaml` ni CLI:
`Blur(p=0.01)`, `MedianBlur(p=0.01)`, `ToGray(p=0.01)`, `CLAHE(p=0.01)`, y además
`RandomBrightnessContrast(p=0.0)`, `RandomGamma(p=0.0)`, `ImageCompression(quality_range=(75,
100), p=0.0)` — estas tres últimas efectivamente desactivadas (`p=0.0`) pero presentes en la
composición. Es idéntica en las cuatro corridas por ser parte fija del código, no de los
hiperparámetros.

> **No aplican a estos experimentos**: `auto_augment` y `erasing` también existen en
> `default.yaml` y aparecen en el log de `args` de cada corrida, pero según su propia
> documentación en el archivo son específicos de **clasificación** — no afectan el pipeline de
> aumento de detección que usan estos experimentos.

Solo controlo parámetros de **ejecución/hardware** (no de red): `epochs`, `imgsz`, `batch`,
`workers`, `amp`, `device` — ver sección 9.

## 7. Metodología de entrenamiento (transfer learning / fine-tuning)

- **Transfer learning + fine-tuning desde pesos preentrenados en COCO, no entrenamiento desde
  cero.** `train_yolo11.py` siempre instancia `YOLO(args.model)` con un checkpoint `.pt`
  (`yolo11n.pt`/`yolo11s.pt`), nunca con un `.yaml` de arquitectura sin entrenar. Verifiqué en
  `ultralytics/engine/trainer.py:803-836` (`BaseTrainer.setup_model`): cuando `self.model` termina
  en `.pt`, llama a `load_checkpoint(self.model)` y usa esos pesos como punto de partida — no hay
  inicialización aleatoria. `pretrained: True` en `ultralytics/cfg/default.yaml:25` confirma la
  intención por defecto, y el script no la sobreescribe. Como VisDrone tiene una sola clase
  (`person`) contra las 80 de COCO, la cabeza de clasificación no calza 1:1 con el checkpoint —
  vas a ver estas dos líneas en el log al arrancar cada corrida, generadas por
  `ultralytics/nn/tasks.py:433` y `:338`:

  ```
  Overriding model.yaml nc=80 with nc=1
  Transferred 319/355 items from pretrained weights
  ```

  (el segundo número es ilustrativo — varía según Nano/Small; lo importante es que **no** dice
  355/355: la cabeza de clasificación se reinicializa para 1 clase, pero el backbone y el cuello
  sí se transfieren completos).

- **Casi ninguna capa congelada — fine-tuning general desde la época 1, con una excepción fija
  por diseño.** `train_yolo11.py` no pasa `freeze=` a `model.train()`, así que se usa el default
  de Ultralytics: `freeze: None` (`ultralytics/cfg/default.yaml:39`, "freeze first N layers") —
  no hay una fase inicial con backbone congelado ni un "unfreeze" progresivo, a diferencia de
  otros flujos de transfer learning (p. ej. clasificación con `torchvision`, donde congelar el
  backbone las primeras épocas es común). Pero **una capa queda siempre congelada sin importar
  `freeze=`**: revisé `ultralytics/engine/trainer.py:337` y el módulo `.dfl` (distribution focal
  loss, la proyección que convierte la distribución de probabilidad de cada borde de caja en una
  coordenada) está hardcodeado en `always_freeze_names = [".dfl"]` — es una proyección fija por
  diseño (no tiene sentido entrenarla), así que el resto de la red sí ajusta el 100% de sus
  parámetros, pero ese módulo puntual no.

- **Hiperparámetros clave del fine-tuning** (ya listados en la sección 6.1; acá el porqué de cada
  uno, y dónde el valor *real* en tiempo de ejecución difiere del que aparece en `default.yaml`
  por el propio comportamiento de `optimizer: auto`):

  | Parámetro | Valor en `default.yaml` | Qué pasa realmente con `optimizer: auto` |
  |---|---|---|
  | `optimizer` | `auto` | Revisé `build_optimizer` en `ultralytics/engine/trainer.py:1094-1122`: con más de 10 000 iteraciones estimadas (`épocas × batches`) resuelve a **`MuSGD`** (Muon-SGD, `lr=0.01, momentum=0.9`); con 10 000 o menos, a **`AdamW`** con `lr0` recalculado (`lr_fit = round(0.002 × 5 / (4 + nc), 6)` — con `nc=1` da `0.002`) y `momentum=0.9`. Esto es distinto de lo que documenta el repo hermano YOLOv12 (`SGD` liso) — esta versión de Ultralytics ya incluye el optimizador Muon |
  | `lr0` | `0.01` | Solo se usa tal cual si terminás fijando un optimizador explícito (no `auto`); con `auto` se recalcula como se explica arriba |
  | `lrf` | `0.01` | Fracción final: la LR decae hasta `lr0 × lrf` al terminar las 250 épocas, sobre el `lr0` que haya quedado vigente |
  | `cos_lr` | `False` | El *scheduler* decae la LR **linealmente**, no con un coseno |
  | `warmup_epochs` | `3.0` | Sin cambios — las primeras 3 épocas interpolan la LR y el momentum en vez de arrancar de golpe |
  | `warmup_momentum` | `0.8` | Sin cambios — el momentum de warmup interpola hacia `self.args.momentum` (`ultralytics/engine/trainer.py:483`), que con `optimizer: auto` **se queda en el `0.937` de `default.yaml`**, no en el `0.9` con el que en realidad se construyó el optimizador (`build_optimizer` no reescribe `self.args.momentum`) — un detalle interno de Ultralytics, no algo que dependa de este script |
  | `warmup_bias_lr` | `0.1` | **Se sobreescribe a `0.0`** en tiempo de ejecución (`ultralytics/engine/trainer.py:1122`, `self.args.warmup_bias_lr = 0.0`) apenas se resuelve `optimizer: auto` — el `0.1` de `default.yaml` nunca llega a aplicarse en las cuatro corridas |

  Esta combinación (fine-tuning general con `.dfl` siempre congelado, warmup de 3 épocas,
  decaimiento lineal, optimizador auto-resuelto) es la misma en las cuatro corridas — Nano y
  Small parten de sus respectivos checkpoints de COCO, no de una arquitectura sin entrenar.

## 8. Credenciales W&B (seguras, sin hardcodear)

- `.env.example` (versionado en git, sin secretos reales) documenta las variables requeridas.
- `.env` (NO versionado, ver `.gitignore`) contiene la `WANDB_API_KEY` real.
- `train_yolo11.py` carga `.env` con `python-dotenv`; si `WANDB_API_KEY` falta, el script aborta
  con un mensaje claro en vez de entrenar sin tracking.
- Las cuatro corridas reportan al **mismo proyecto W&B** (`WANDB_PROJECT`, p. ej. `YOLOv11`);
  modelo (Nano/Small) y dataset (Base/Augmented) se distinguen por el **nombre de la corrida**
  (`--name nano_base`, `nano_augmented`, `small_base`, `small_augmented`), no por el proyecto.

> **Por qué el script no llama a `wandb.login(key=...)`**: en algunas versiones de `wandb`, esa
> función valida que la key tenga exactamente el formato clásico de key personal (longitud fija)
> y puede rechazar keys con prefijo más largas (p. ej. de cuentas de servicio/organización) aunque
> sean completamente válidas — revisé el código de validación instalado en este entorno
> (`wandb/sdk/lib/wbauth/validation.py`, `wandb==0.28.2`) y esa versión concreta ya lo maneja bien
> (exige un mínimo de caracteres, no una longitud exacta, y reconoce el formato con prefijo), pero
> prefiero no depender de eso. El script exporta `WANDB_API_KEY` al entorno vía `python-dotenv` y
> deja que `wandb.init()` la tome directamente, sin pasar por `wandb.login()` — funciona igual sin
> importar la versión de `wandb` instalada.
>
> **Por qué el script llama a `wandb.init()` explícitamente antes de entrenar**: el callback
> nativo de Ultralytics (`ultralytics/utils/callbacks/wb.py`) deriva el nombre de proyecto de W&B
> a partir del `project=` que se le pasa a `model.train()` — que, si no se maneja con cuidado,
> termina siendo una ruta local de carpeta (con `\` y `:` en Windows). Ese callback solo limpia el
> carácter `/`, no `\` ni `:`, así que en Windows terminaría pasándole a W&B un nombre de proyecto
> inválido y W&B lo rechazaría con `UsageError: Invalid project name '...': cannot contain
> characters '/,\,#,?,%,:'`. El fix es inicializar W&B antes de `model.train()`, con
> `project=WANDB_PROJECT` (el proyecto único y limpio, sin caracteres de ruta) y `name=` la
> corrida — el callback nativo detecta que ya hay un run activo (`wb.run`) y solo loguea métricas
> en él, sin volver a llamar a `wb.init()`. De paso, `model.train()` ya **no** recibe `project=`,
> así que los resultados locales quedan en `runs/detect/<name>/`, el default de Ultralytics,
> independiente del proyecto de W&B.

## 9. Ejecutar las cuatro corridas (PowerShell)

Con el entorno `yolov11` (sección 3) activado, ejecuta las cuatro corridas — Nano y Small, cada
uno con Base y Augmented:

```powershell
# Copia la plantilla de credenciales y completa tu WANDB_API_KEY/WANDB_PROJECT reales
Copy-Item .env.example .env
notepad .env

# Nano — dataset base
python train_yolo11.py `
    --data data\visdrone_base.yaml `
    --name nano_base `
    --model yolo11n.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Nano — dataset aumentado (offline)
python train_yolo11.py `
    --data data\visdrone_augmented.yaml `
    --name nano_augmented `
    --model yolo11n.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Small — dataset base
python train_yolo11.py `
    --data data\visdrone_base.yaml `
    --name small_base `
    --model yolo11s.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Small — dataset aumentado (offline)
python train_yolo11.py `
    --data data\visdrone_augmented.yaml `
    --name small_augmented `
    --model yolo11s.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8
```

Los resultados locales quedan en subcarpetas independientes: `runs/detect/nano_base/`,
`runs/detect/nano_augmented/`, `runs/detect/small_base/` y `runs/detect/small_augmented/` (ya
cubiertas por la regla `runs/` del `.gitignore`). En W&B, las cuatro corridas caen en el mismo
proyecto (`WANDB_PROJECT`), distinguidas por nombre de corrida.

> **Resolución de imagen (`imgsz=640`, el estándar YOLO)**: en modo `train`, Ultralytics recibe
> `imgsz` como un único entero que define el lado largo del letterbox cuadrado; el lado corto se
> rellena (padding) en vez de recortarse o deformarse. `640` es el valor de
> `ultralytics/cfg/default.yaml` y con el que Nano/Small fueron ajustados originalmente en COCO.
> `batch=16` es razonable a `640px` para modelos de este tamaño en 16 GB de VRAM — Nano (2.6M
> parámetros, 6.5 GFLOPs a 640px) y Small (9.4M / 21.5 GFLOPs) son mucho más livianos que Medium
> (20.1M / 68 GFLOPs) o Large (25.3M / 86.9 GFLOPs), que fue justo lo que motivó bajar de tamaño
> de modelo (ver el aviso al inicio de este documento). No lo tengo verificado todavía
> específicamente en este dataset: si aparece `OOM` baja `--batch` (o usa `--batch -1` para
> AutoBatch, que mide la VRAM libre real), y si sobra VRAM puedes subirlo — usa siempre el
> **mismo** valor en las cuatro corridas para que la comparación siga siendo válida.

> **"El entrenamiento es muy lento / no avanza" (causa raíz posible, verificada en el código de
> este repo)**: si aparece `WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU`
> justo al arrancar una época, **esa es la causa**, no un cuelgue —
> `ultralytics/utils/tal.py:104` atrapa el `OutOfMemoryError` en ese paso puntual y hace fallback
> silencioso a CPU (mueve los tensores GPU→CPU, calcula ahí, los regresa a GPU) **en cada
> iteración**, lo que hace que la GPU se vea casi al límite de uso pero el entrenamiento avance
> extremadamente lento. VisDrone tiene muchísimas cajas por imagen, lo que infla el tensor de
> costo de asignación independientemente del tamaño del modelo — por eso puede pasar incluso con
> Nano/Small si `--batch`/`--imgsz` quedan altos para tu VRAM. Si ves ese warning, baja `--batch`
> y/o `--imgsz`, o usa `--batch -1`. Conviene revisar el `s/it`/`ETA` de la barra de progreso antes
> de asumir un cuelgue. Si aparece `BrokenPipeError`/`EOFError` (multiprocessing en Windows), baja
> `--workers` a `0`.
>
> A diferencia de YOLOv12 (bloques `A2C2f` / Area Attention, con `flash-attn` como dependencia
> opcional), la arquitectura C3k2/C2PSA de YOLO11 no referencia `flash-attn` ni imprime ningún
> aviso de "FlashAttention is not available" en ningún punto de este repo — verificado buscando en
> todo `ultralytics/`. Si el entrenamiento va lento en YOLO11, no es por eso.

### Otros tamaños de modelo (Medium, Large — no usados actualmente)

Fuera del alcance actual del proyecto (ver aviso al inicio del documento). El script no cambió —
`--model`, `--data` y `--name` siguen siendo parámetros de línea de comandos, así que retomar
Medium o Large no requiere tocar código, solo estos comandos de referencia:

```powershell
# Medium — dataset base
python train_yolo11.py `
    --data data\visdrone_base.yaml `
    --name medium_base `
    --model yolo11m.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8

# Medium — dataset aumentado (offline)
python train_yolo11.py `
    --data data\visdrone_augmented.yaml `
    --name medium_augmented `
    --model yolo11m.pt `
    --epochs 250 `
    --imgsz 640 `
    --batch 16 `
    --workers 8
```

> Con `yolo11l.pt` (Large) o a `imgsz` mayor a `640`, baja `--batch` con margen (o usa
> `--batch -1`) — cuanto más grande el modelo y mayor la resolución, más cerca del límite de 16 GB
> de VRAM, y VisDrone agrava el riesgo de OOM en `TaskAlignedAssigner` (ver nota más arriba).

## 10. Métricas registradas en W&B

La integración nativa de Ultralytics (`ultralytics/utils/callbacks/wb.py`, activada vía
`settings.update({"wandb": True})`) ya registra automáticamente, por época:

- `metrics/precision(B)`, `metrics/recall(B)`
- `metrics/mAP50(B)`, `metrics/mAP50-95(B)`
- Pérdidas de entrenamiento: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`
- Curvas Precision-Recall, F1-Confidence (una serie por clase; acá solo `person`)
- Matriz de confusión y artefacto del mejor checkpoint (`best.pt`)

`train_yolo11.py` agrega un callback adicional (`log_person_metrics`, en `on_fit_epoch_end`) que
registra las mismas métricas con nombres explícitos bajo el prefijo `person/` para lectura
directa en el dashboard:

- `person/precision`, `person/recall`, `person/f1_score`
- `person/mAP50`, `person/mAP50-95`
- `person/iou_at_0.5` (= `mAP50`: fracción de detecciones con IoU ≥ 0.5, la definición operativa
  de "IoU" a nivel de dataset en detección de objetos — no existe un IoU escalar único por época
  en detección, a diferencia de segmentación)
- `person/accuracy` (índice de Jaccard `TP / (TP + FP + FN)` derivado de
  `trainer.validator.metrics.confusion_matrix.matrix`, una matriz `2×2` `person` vs. `background`
  con `nc=1`; es la métrica más cercana a "accuracy" en detección de un solo objeto, ya que no
  existe accuracy de clasificación estándar cuando no hay negativos verdaderos explícitos por
  imagen)

## 11. Verificación de GPU (incluida en el script)

Antes de cada entrenamiento, `train_yolo11.py` imprime y valida:

```python
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)
```

Si `torch.cuda.is_available()` es `False`, el script aborta con `RuntimeError` antes de cargar
credenciales de W&B o el dataset.

> **No es un error, es el chequeo de AMP**: justo después, ya con `amp=True` (fijo en el script),
> Ultralytics corre su propio chequeo automático de precisión mixta
> (`ultralytics/engine/trainer.py:359-363` llamando a `check_amp()` en
> `ultralytics/utils/checks.py:976`) antes de cada una de las cuatro corridas. Ese chequeo
> descarga y corre inferencia FP32 vs. FP16 sobre un modelo nano de referencia — en esta versión
> del repo es **`yolo26n.pt`** (`ultralytics/utils/checks.py:1030`), no `yolo11n.pt` ni el
> `--model` que le pasaste — para confirmar que AMP no produce NaN o mAP en cero en tu GPU. Es
> independiente de `--model`: pasa igual entrenando Nano o Small, y tu modelo real es el que sí se
> usa para entrenar. El mensaje `AMP: running Automatic Mixed Precision (AMP) checks... ✅` en el
> log es justo este paso. Se descarga una sola vez y queda cacheado en la carpeta desde donde
> corres el script; si se repite en cada corrida, revisa que estés lanzando `train_yolo11.py`
> siempre desde el mismo directorio.

## 12. Resolución de problemas

Problemas reales que encontré y resolví durante estas pruebas, en el orden en que suelen
aparecer. Cada fila tiene la explicación completa en la sección indicada.

| Síntoma | Causa | Fix | Detalle |
|---|---|---|---|
| `torch.cuda.is_available()` da `True` pero el entrenamiento falla o cae a CPU sin avisar | El wheel de PyTorch instalado (`cu124`) no incluye kernels para Blackwell (`sm_120`, RTX 50-series) | Reinstalar con `--index-url https://download.pytorch.org/whl/cu128` | sección 1, sección 4 |
| `pip install -r requirements-windows.txt` falla con `Building wheel for stringzilla ... Microsoft Visual C++ 14.0 or greater is required` | `albumentations` arrastra `albucore`→`stringzilla>=3.10.4`, que no publica wheel para Windows desde su serie 2.x | Instalar *Build Tools for Visual Studio* y marcar explícitamente el workload **"Desktop development with C++"** (el instalador base solo, sin ese workload, no basta) | sección 2 |
| `wandb.errors.UsageError: Invalid project name '...': cannot contain characters '/,\,#,?,%,:'` | El callback nativo de Ultralytics derivaría el nombre de proyecto de W&B a partir de una ruta local de Windows (con `\` y `:`) si se le pasa `project=` a `model.train()` | El script llama a `wandb.init(project=..., name=...)` con el nombre de proyecto limpio antes de `model.train()`, y no le pasa `project=` a `model.train()` | sección 8 |
| GPU casi al 100% de uso pero el entrenamiento no avanza (época pegada) | `WARNING: CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU` — `batch`/`imgsz` demasiado altos para la VRAM disponible con este dataset (VisDrone tiene muchísimas cajas por imagen) | Bajar `--batch` y/o `--imgsz`, o usar `--batch -1` (AutoBatch) | sección 9 |
| Se descarga `yolo26n.pt` al arrancar cada corrida, aunque uses `--model yolo11n.pt`/`yolo11s.pt` | Chequeo automático de AMP de Ultralytics, no un error ni algo específico de tu `--model` | Nada que arreglar — se cacheará tras la primera descarga si siempre lanzas el script desde el mismo directorio | sección 11 |
