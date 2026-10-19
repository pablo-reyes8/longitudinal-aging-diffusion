# Mapa del proyecto: `aging-difussion-supervised`

Este documento resume la estructura funcional del proyecto para poder ubicarse
rápidamente en el código. Se revisaron `src/`, `data/` y la documentación
existente en `docs/`. No es un inventario exhaustivo de cada archivo ni cubre
tests u otros directorios.

## 1. Qué hace el proyecto

Es un sistema de edición de rostros condicionada por edad. Parte de pares
longitudinales de la misma persona (`imagen origen`, `imagen objetivo`,
`edad origen`, `edad objetivo`) y adapta un backbone de Stable Diffusion para
producir una versión más joven o más vieja del rostro.

El proyecto cubre el ciclo completo:

1. Indexar imágenes por identidad y edad.
2. Construir pares longitudinales y `DataLoader` sin fuga de identidad entre
   train/validation/test.
3. Construir un bundle de difusión con condicionamiento de imagen y edad.
4. Entrenar adaptadores LoRA/DoRA y un condicionador numérico de edad.
5. Inferir ediciones directas o mediante inversión DDIM.
6. Monitorear edad, identidad y transiciones durante el entrenamiento.
7. Evaluar checkpoints con AdaFace, DEX y KID.
8. Comparar contra modelos externos y ejecutar ablaciones controladas.

## 2. Flujo mental del sistema

```text
data/ (imágenes)
    ↓
manifiestos → splits por identidad → pares source/target → DataLoaders
    ↓
src/model/  (Stable Diffusion + fuente + edad + LoRA/DoRA)
    ↓                         ↘
src/loss/  (difusión + edad + identidad)   src/inference/
    ↓                                      ↓
src/training/                         imágenes generadas
    ↓                                      ↓
checkpoints + monitoreo             src/quantitative_metrics/
                                           ↓
                                  métricas y reportes

src/ablation_studies/ reutiliza entrenamiento + inferencia + métricas
src/comparisong/ compara contra baselines externos y el modelo propio
```

## 3. Puntos de entrada públicos

Estos son los símbolos más útiles para orientarse sin leer toda la base de
código:

```python
from data import build_face_aging_dataloaders
from src.model import build_face_aging_diffusion_bundle
from src.loss import FaceAgingDiffusionLoss
from src.training import train_model
from src.inference import infer_face_aging
from src.comparisong import load_aging_models, age_image
from src.quantitative_metrics import evaluate_aging_outputs
from src.ablation_studies import ablation_studies
```

En `src/training/__init__.py` también existe el alias histórico
`TRAIN_AGGING_MODEL` además de `TRAIN_AGING_MODEL`.

## 4. `data/`: preparación de datos longitudinales

Esta carpeta contiene el pipeline de datos, no el modelo. La convención
principal es tener carpetas por identidad con archivos cuyo nombre empieza por
la edad, por ejemplo `35.jpg` o `35_1.jpg`.

### Archivos principales

- `data/indexing.py`: define `ImageRecord` y `PairRecord`; escanea carpetas,
  parsea edades, normaliza etiquetas sospechosas, crea manifiestos y asigna
  identidades completas a `train`, `val` y `test`.
- `data/dataset.py`: implementa `FaceAgingDataset`, selección determinista de
  targets, pares forward/reverse, pares de delta cero, flips y la colación de
  batches. Devuelve imágenes normalizadas a `[-1, 1]`, edades, prompts,
  identidad y paths.
- `data/dataloaders.py`: fábrica de los tres `DataLoader`. También puede
  combinar el dataset primario con FG-NET/Kaggle y devuelve metadata del
  pipeline.
- `data/fgnet.py`: interpreta nombres como `001A43a.JPG`, crea un manifiesto
  compatible con el pipeline general y selecciona pares complementarios de
  FG-NET procurando cubrir transiciones poco representadas.
- `data/prompts.py`: construye los prompts de origen y objetivo a partir de las
  edades y el estilo elegido (`selfage` o `fading`).
- `data/validation.py`: audita fugas de identidad, paths duplicados, pares,
  prompts, imágenes corruptas, distribución de edades y concentración por
  identidad.
- `data/diagnostics.py`: utilidades pequeñas para inspeccionar un batch y
  visualizar una cuadrícula source/target.

### Datos incluidos en el repositorio

- `data/sample/`: 18 imágenes distribuidas entre `id_1144`, `id_1145` e
  `id_1146`; sirve como muestra longitudinal organizada por identidad.
- `data/sample_colombia/`: 8 imágenes de ejemplo en formato plano.
- `data/sample_kaggle/`: 31 imágenes con nombres de estilo FG-NET, como
  `001A02.JPG` y `001A43a.JPG`.
- `data/__init__.py`: reexporta la API del pipeline.

Los datasets completos no parecen vivir dentro de `data/`; el loader recibe
sus rutas mediante `build_face_aging_dataloaders(root_dir, ...)` y puede usar
una ruta externa para FG-NET/Kaggle.

## 5. `src/model/`: backbone y condicionamiento

Aquí se construye el bundle de difusión y se define qué parámetros se
entrenan.

- `src/model/load_diffusion_models.py`: módulo central. Carga los componentes
  de Stable Diffusion 1.x, amplía la entrada del U-Net para concatenar la
  imagen fuente con el target ruidoso, instala el condicionador de edad,
  congela lo que corresponde, agrupa parámetros del optimizador y guarda/carga
  adaptadores.
- `src/model/age_conditioning.py`: implementa `AgeDeltaConditioner` y
  `AgeConditionerV2`. El segundo usa edad origen, edad objetivo y delta con
  features de Fourier; ambos inyectan el vector proyectado en el embedding de
  tiempo del U-Net.
- `src/model/LoRa.py`: implementación manual de LoRA y descubrimiento de
  módulos de atención (`to_q`, `to_k`, `to_v`, `to_out.0`).
- `src/model/DoRa.py`: implementación manual de DoRA usando la misma selección
  de capas que LoRA.
- `src/model/smoke_forward_models.py`: preparación y validación de un forward
  de modelo para detectar problemas estructurales antes de entrenar.
- `src/model/__init__.py`: API pública de construcción, checkpoints, latentes,
  prompts y adaptadores.

La configuración canónica de ablaciones usa DoRA, el condicionador V2 y una
entrada del U-Net con la imagen fuente concatenada al target.

## 6. `src/loss/`: objetivo de entrenamiento

La pérdida combina el objetivo de difusión con señales auxiliares congeladas:

- `src/loss/face_aging_loss.py`: `FaceAgingDiffusionLoss`, pérdida compuesta
  con término de difusión, identidad, edad absoluta, edad relativa,
  preservación para deltas pequeños y ponderaciones especiales.
- `src/loss/diffusion_utils.py`: targets `epsilon`/`v_prediction`, coeficientes
  del scheduler, reconstrucción de `x0`, Min-SNR y conversiones de rango.
- `src/loss/auxiliary_adapters.py`: adapta redes externas para producir
  embeddings de identidad normalizados y edad continua; contiene la pérdida de
  similitud coseno de identidad.
- `src/loss/pretrained_auxiliary.py`: carga y adapta ArcFace y MiVOLO como
  modelos congelados diferenciables para las pérdidas auxiliares.
- `src/loss/validation.py`: validaciones algebraicas, estabilidad numérica,
  flujo de gradientes y descomposición de la pérdida.
- `src/loss/__init__.py`: reexporta la API del objetivo supervisado.

## 7. `src/training/`: entrenamiento de un modelo

### Orquestación y ciclo principal

- `src/training/train_face_aging.py`: entrada de alto nivel `train_model`. Une
  bundle, pérdida, optimizador, scheduler, AMP, acumulación de gradiente,
  validación, checkpoints, monitoreo y resume.
- `src/training/train_one_epoch.py`: recorre batches, ejecuta acumulación de
  gradientes, clipping y pasos del optimizador.
- `src/training/training_step.py`: prepara latentes, ruido, timesteps,
  dropout de condicionamiento, forward del U-Net y cálculo de la pérdida.
- `src/training/validate_one_epoch.py`: evaluación determinista del modelo en
  validation.

### Soporte de entrenamiento

- `src/training/checkpoints.py`: payloads de entrenamiento/inferencia,
  compatibilidad de configuración, guardado atómico y
  `TrainingCheckpointManager` (`latest`, `best` y snapshots por época).
- `src/training/mixed_precision.py`: selección de device, AMP, scaler,
  movimiento de batches y paso seguro del optimizador.
- `src/training/scheduler_warmup.py`: warmup + cosine decay y estimación de
  pasos.
- `src/training/timestep_sampling.py`: muestreo uniforme y timesteps
  deterministas de validación.
- `src/training/conditioning_dropout.py`: dropout independiente de texto,
  imagen y ambas condiciones.
- `src/training/prompt_regularization.py`: política de prompts numéricos o
  genéricos durante entrenamiento.
- `src/training/metrics.py`, `seed.py`: acumuladores de métricas, learning
  rates y reproducibilidad.

### Monitoreo y diagnósticos

- `src/training/sampling_monitor.py`: genera ediciones/sweeps fijos por época,
  guarda imágenes y CSV de diagnósticos de edad/identidad.
- `src/training/age_calibration.py`: calibra la respuesta de edad y resume
  métricas direccionales forward/reverse.
- `src/training/validation_training_pipeline.py`: preflight reproducible de
  timesteps, dropout, optimizer, memoria y smoke tests.

## 8. `src/inference/`: generación y análisis de ediciones

- `src/inference/infer_face_aging.py`: API principal `infer_face_aging` para
  edición directa o inversa; resuelve strength, prompts, CFG, condicionamiento
  de edad y devuelve imagen, latentes y metadata. También ofrece wrappers
  direct/inverse y strength adaptativo.
- `src/inference/inference_utils.py`: carga/preprocesamiento de imágenes,
  encode/decode VAE, scheduler de inferencia, dtype/device, seeds y conversión
  a PIL.
- `src/inference/ddim_inversion.py`: inversión DDIM determinista y edición a
  partir del latent invertido.
- `src/inference/cfg_guidance.py`: combina las ramas de guidance de texto,
  imagen y edad.
- `src/inference/prompt_building.py` y `prompt_assistance.py`: construyen
  prompts de inferencia y variantes asistidas.
- `src/inference/checkpoint_loading.py`: reconstruye el bundle y carga un
  checkpoint de adaptador con validación estricta de configuración.
- `src/inference/source_image_loading.py`: acepta paths/URLs y aplica las
  reglas de resolución y preprocesamiento para sweeps.
- `src/inference/diagnostics.py` y `delta_bin_evaluation.py`: edad/identidad
  por muestra y agregación por bins de delta de edad.
- `src/inference/comparison_helpers.py`: genera age sweeps, strength sweeps y
  cuadrículas de comparación.
- `src/inference/checkpoint_diagnostics.py`: compara múltiples checkpoints con
  sweeps de edad, strength y strength adaptativo.
- `src/inference/conditioning_isolation.py`: analiza cuánto aportan prompt,
  imagen y condicionamiento de edad.
- `src/inference/smart_age_sweep.py`: prueba varios candidatos por edad y
  selecciona el mejor según error de edad, confianza e identidad.
- `src/inference/validation_inference_pipeline.py`: validación estructurada de
  la ruta de inferencia.

## 9. `src/quantitative_metrics/`: evaluación independiente

Este paquete está separado del entrenamiento y de la selección de candidatos.
La unidad de trabajo es un `prediction_manifest` con paths, edades, identidad
del modelo y nombre del modelo.

- `src/quantitative_metrics/contracts.py`: contrato y validación del manifest;
  conserva filas aunque haya fallos.
- `src/quantitative_metrics/backends.py`: backends congelados de AdaFace, DEX
  y KID, incluyendo alineamiento, embeddings de identidad y edad esperada.
- `src/quantitative_metrics/evaluator.py`: evaluación por muestra y agregada,
  cache por SHA-256, fallos y persistencia de reportes.
- `src/quantitative_metrics/assets.py`: prepara/valida los checkpoints y el
  repositorio de AdaFace; usa rutas locales una vez preparados los assets.
- `src/quantitative_metrics/image_io.py`: resolución de imágenes locales o
  URLs y hashing de bytes.
- `src/quantitative_metrics/integration.py`: puente genérico
  inferencia → imágenes generadas → manifest → evaluator.
- `src/quantitative_metrics/api.py`: helpers de más alto nivel para evaluar un
  checkpoint, una lista de edades, KID y árboles de imágenes etiquetadas.
- `src/quantitative_metrics/__init__.py`: API pública del evaluador.

Las métricas principales previstas por la documentación son AdaFace
ID-Source, AdaFace ID-GT, DEX Age MAE y KID. Los reportes escriben manifest,
métricas por muestra, resumen CSV/JSON, log de fallos y metadata de
reproducibilidad.

## 10. `src/comparisong/`: baselines externos y comparación

El nombre de la carpeta es `comparisong`, y se usa como paquete público de
comparaciones contra modelos externos.

- `src/comparisong/load_models.py`: prepara los baselines SAM, FRAN, FADING,
  CUSP, HRFAE y Cradle2Cane, con manejo de repositorios, checkpoints,
  memoria y, opcionalmente, el bundle de métricas.
- `src/comparisong/generate_images.py`: fachada `age_image`; normaliza la
  entrada, ejecuta el modelo seleccionado y guarda resultados en una
  convención común.
- `src/comparisong/evaluate_test_set.py`: crea transiciones longitudinales,
  ejecuta los modelos y evalúa todos sobre un panel compartido; soporta
  reanudación, fallos por muestra y variantes del modelo propio.
- `src/comparisong/resident_models.py`: mantiene un solo baseline pesado
  residente por fase y controla su colocación en CPU/GPU.
- `src/comparisong/resident_worker.py`: worker aislado para runners externos
  de FADING/HRFAE, evitando colisiones de imports y recargas costosas.
- `src/comparisong/identity_metadata.py`: etiquetas de género suministradas
  para identidades del set de comparación.
- `src/comparisong/__init__.py`: expone `load_aging_models`, `age_image`,
  `evaluate_test_set` y `select_test_transitions`.

## 11. `src/ablation_studies/`: experimentos controlados

Reutiliza los mismos loaders, bundle, entrenamiento, inferencia y métricas para
comparar mecanismos concretos con una configuración canónica.

- `src/ablation_studies/config.py`: define casos 0–5 y crea copias aisladas de
  la configuración. Los casos eliminan, respectivamente, condicionamiento
  numérico, adaptador, pérdida de edad relativa, preservación/deltas cero o
  pérdida de identidad.
- `src/ablation_studies/panel.py`: crea y persiste un panel longitudinal fijo;
  rechaza que distintas corridas usen pares diferentes.
- `src/ablation_studies/evaluation.py`: evalúa el panel después de cada época y
  al final, escribiendo manifests y resúmenes.
- `src/ablation_studies/runner.py`: orquesta casos, datos, modelos, métricas,
  entrenamiento, resume, aislamiento de salidas y metadata del experimento.
- `src/ablation_studies/__init__.py`: API pública para configuración, panel y
  `ablation_studies(...)`.

## 12. Qué leer primero según la tarea

| Necesidad | Orden recomendado |
|---|---|
| Entender los datos | `data/indexing.py` → `data/dataset.py` → `data/dataloaders.py` |
| Entender qué se entrena | `src/model/load_diffusion_models.py` → `src/model/age_conditioning.py` → `src/model/DoRa.py` |
| Entender la pérdida | `src/loss/face_aging_loss.py` → `src/loss/diffusion_utils.py` → `src/loss/auxiliary_adapters.py` |
| Seguir un entrenamiento | `src/training/train_face_aging.py` → `train_one_epoch.py` → `training_step.py` → `checkpoints.py` |
| Generar una imagen | `src/inference/infer_face_aging.py` → `inference_utils.py` → `ddim_inversion.py` |
| Interpretar resultados | `src/inference/diagnostics.py` → `src/quantitative_metrics/evaluator.py` |
| Comparar modelos | `src/comparisong/load_models.py` → `generate_images.py` → `evaluate_test_set.py` |
| Entender ablaciones | `src/ablation_studies/config.py` → `panel.py` → `runner.py` |

## 13. Documentación existente relacionada

- `docs/superpowers/specs/2026-09-26-quantitative-evaluation-design.md`:
  propósito y contratos científicos del evaluador AdaFace/DEX/KID.
- `docs/superpowers/plans/2026-09-26-quantitative-evaluation.md`: plan de
  implementación e integración de métricas con comparación e inferencia.
- `docs/superpowers/plans/2026-09-26-training-ablation-runner.md`: plan de
  configuración, panel fijo, evaluación por época y runner de ablaciones.

## 14. Resumen en una frase por familia

- `data/`: convierte fotos etiquetadas por identidad/edad en pares y batches.
- `src/model/`: convierte Stable Diffusion en un editor condicionado por fuente
  y edad.
- `src/loss/`: enseña difusión, edad y conservación de identidad.
- `src/training/`: coordina optimización, validación, checkpoints y monitoreo.
- `src/inference/`: produce ediciones y diagnósticos sobre checkpoints.
- `src/quantitative_metrics/`: mide resultados con un evaluador independiente.
- `src/comparisong/`: ejecuta y compara baselines externos.
- `src/ablation_studies/`: mide qué aporta cada mecanismo del sistema.
