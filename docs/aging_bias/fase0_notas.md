# Evaluación de sesgo v2 — límites y reutilización

La especificación es `planing/GUIA_AGENTE_metrica_sesgo_envejecimiento.md`.
No se modifica `src/inference`, `src/model`, `src/ablation_studies` ni las APIs
legacy. Los nuevos módulos se importan explícitamente, nunca desde sus init.

- Se reutilizan `ImageResolver` (SHA-256 de bytes + EXIF), el contrato del
  manifest y los backends AdaFace/KID/DEX existentes, sin cambiar sus resultados.
- Las imágenes ya generadas viven en
  `/content/drive/MyDrive/respaldo_test_comparison/generated/{modelo}/{identity_id}/{sample_id}.png`.
  El manifest local está en `output/comparsing_metrics/output/`. No contiene
  imágenes; no se regenera el panel para recalcular métricas.
- La nueva evaluación acepta `prediction_manifest.csv` o `transitions.csv`
  y raíces reubicadas. Guarda firmas de contenido/configuración y caché por
  imagen; no confía en rutas absolutas antiguas de Colab.
- La generación sigue usando `load_aging_models(..., load_metrics=False)` y
  sesiones residentes; el bundle de evaluación nuevo es independiente y CPU.
- Colombia se indexa con `data.indexing.build_image_manifest`, conservando
  fotografías distintas a la misma edad para ICC. Se exige identificación
  explícita de train y exclusión de IDs test. `sample_kaggle` NO sustituye a
  FFHQ-Aging: es otro dataset y no el proxy poblacional especificado.
- La regresión usa atributos clásicos, within-FE y WLS; puede ser no
  identificable. Nunca se fuerzan razones a [0,1] ni se inventan curvas/ICC.
  Las referencias transversales y longitudinales no son intercambiables.
- Edad-período-cohorte sigue siendo una limitación: referencias compartidas
  facilitan comparar modelos, pero no eliminan error de especificación común.
- No se descargan pesos ni se ejecutan redes durante implementación/tests.
  Los notebooks son los puntos de ejecución real. La revisión de anclas reales
  es obligatoria antes del experimento contrafactual SD1.5.

## Fuentes verificadas (2026-10-18)

FFHQ-Aging: https://github.com/royorel/FFHQ-Aging-Dataset
FairFace: https://github.com/dchen236/FairFace/blob/master/predict.py
FRAN reproduction utilizada: https://github.com/timroelofs123/face_reaging
FADING entrenamiento: https://github.com/MunchkinChen/FADING
Las etiquetas y convenciones se documentan junto a sus adaptadores; lo no
verificado se registra como `unknown`, no como ausencia de contaminación.
