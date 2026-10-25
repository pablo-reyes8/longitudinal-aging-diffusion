# Cierre versionado del módulo 1

La especificación es `planing/GUIA_AGENTE_modulo1_cierre.md`. No se cambia
inferencia, entrenamiento, ablations ni la firma de `build_reference_bundle`.
El notebook activa `ReferenceConfig.module1_close=True`; las opciones nuevas
están desactivadas en las configuraciones públicas antiguas. El hash de
`FeatureStore` conserva exactamente el esquema antiguo cuando están apagadas.

## Rutas y ejecución

- `REFERENCE_DIR`: destino NUEVO, por defecto `aging_bias_references_v2`.
- `PREVIOUS_REFERENCE_DIR`: opcional y de solo lectura; no se escribe ni dentro
  de esa carpeta. La API directa puede recuperar su configuración si no recibe
  `reference_config`; en el notebook prevalece la configuración explícita.
- Un bundle antiguo en el destino provoca error, no sobrescritura.
- Todas las detecciones, cachés, parquet, tablas y figuras nuevas van al destino
  nuevo. DINO ocupa su subcarpeta `dinov2`; no mueve el encoder por imagen.
- `close_module1(Module1CloseConfig(...), feature_backend=..., landmark_backend=...)`
  devuelve bundle, DataFrames y rutas. El constructor permite desactivar cada
  reporte o transformación para diagnósticos. La llamada antigua del notebook
  delega al cierre y sigue devolviendo `ReferenceBundle`.

## Medición y límites

Solo se añaden dos atributos principales: `geom_malar_descent` y
`geom_midface_height`. `geom_malar_descent_rel` es un campo de diagnóstico, no
un tercer atributo principal. Índices candidatos MediaPipe:

| Punto | Índices |
| --- | --- |
| Malar | 116 / 345 |
| Comisuras | 61 / 291 |
| Bicigomático | 234 / 454 |

Son posiciones de malla, **no una localización clínica validada del pómulo**.
La primera foto real disponible genera `figures/malar_landmarks_check.png` al
correr Colab. Revisa esa figura antes de interpretar estos atributos. La
verificación real no se ejecutó localmente: `closure_status.json` deja
`landmarks_reviewed=false`; no se inventa una imagen de control ni se afirma
validación anatómica por pasar pruebas geométricas.

La tabla de espejo se limita a los índices efectivamente usados, no pretende
mapear toda la malla. Parejas: 33↔263,160↔387,158↔385,133↔362,153↔380,
144↔373; 70↔300,63↔293,105↔334,66↔296,107↔336;
61↔291,40↔270,0↔0,321↔91,17↔17; 116↔345,234↔454.
El contorno mandibular se invierte, el punto 152 es fijo. Los tests comprueban
involución, reflexión y similitudes. TTA promedia mediciones con esta reflexión;
para descriptores ya simétricos puede no mejorar nada: se reporta el ICC real,
no se presupone reducción del ruido.

El objetivo de resolución es el percentil 10 de IOD original válido del train
latam (nunca FFHQ/test). Se degrada con reducción de área y retorno bicúbico al
grid canónico; no se añade degradación a imágenes inferiores al objetivo.
El grid de alineación no crea detalle. El objetivo queda congelado en
`FeatureConfig.target_iod_px` para aplicar lo mismo en evaluación. La sensibilidad
sin igualación usa las escalas del train igualado para atributos compartidos y
mantiene fija la transformación de zonas.

FairFace permanece transversal, con sus edades estimadas originales. No se
vuelven a estimar y no participa en ICC ni en el ajuste within-person. Sus
identidades son singletons. Las etiquetas Latino_Hispanic no prueban
nacionalidad colombiana ni representatividad regional.

Las zonas usan componentes activos con ICC ≥ .30, pesos ICC y signos del
longitudinal colombiano 30→70. Una orientación no identificable excluye el
componente explícitamente; no se decide mirando FFHQ. El índice tiene ICC propio
y requiere todos sus componentes para evitar reponderación distinta por foto.
Homogeneización conserva ICC ≥ .50. No se garantiza que las zonas mejoren ICC.

FFHQ 40+ utiliza todos los archivos elegibles del CSV local. Con el chunk de
descarga del usuario eso significa todas las fotos de las partes descargadas,
no las 70.000 fotos ni archivos ajenos al CSV. En los jóvenes se conserva el tope.

### Estimación adicional: SOLO FFHQ 50–69

Las etiquetas oficiales no separan 50–59/60–69. Para figuras se ejecuta el
FairFace ResNet-34 ya soportado, únicamente sobre FFHQ `age_group=50-69`, en
lotes, preferentemente CUDA. No se cargan AdaFace, DEX, KID ni MiVOLO. El
checkpoint existente se reutiliza; descargas solo cuando el usuario ejecuta el
notebook con `module1_allow_download=True`. La API acepta `age_backend` o un
checkpoint local para no depender de la descarga. Valida el backend antes de
la extracción larga. DINO y el estimador no se recargan por imagen.

`age`/`age_group` oficiales no cambian. Se guardan `mean_face_age`,
`age_pred_raw`, procedencia y estado. Predicciones fuera de [50,70) se reportan,
no se recortan ni se rellenan. Los demás grupos usan la convención oficial.
Las referencias/contrastes principales siguen usando etiquetas oficiales.
Las figuras finas son **descriptivas y parcialmente asistidas por estimador**,
no edades reales. Se conserva además una figura con bandas comunes 30–39/50–69.

## Salidas

- `diagnostics/icc_tta_comparison.csv`: ICC medido sin/con TTA, zonas incluidas.
- `diagnostics/ffhq_age_coverage.csv`: elegibles, seleccionadas, extracción y QC.
- `diagnostics/resolution_matching.csv`, `resolution_sensitivity.csv`.
- `diagnostics/ffhq_50_69_age_predictions.csv`, `mean_face_coverage.csv`.
- `population_contrasts.csv`: aliases antiguos y `difference_long/cross`, IC
  por identidad latam y por imagen dentro de grupo FFHQ. Condicionados a escalas,
  pesos/signos entrenados y selección de atributos; no son IC conjuntos de todas
  las etapas de selección/preprocesamiento.
- `tables/zone_consistency.csv`: regla fija ≥2 de 3 transiciones por diseño,
  acuerdo de signos; no equivale a significación. `direction` se orienta a
  envejecimiento con el signo congelado; diferencias numéricas quedan originales.
- `figures/mean_faces_grid.png`, `aging_difference_maps.png`,
  `aging_difference_maps_common_50_69.png`, `face_zones.png` y `face_zones_es.png`.
- `internal/placebo_ffhq.csv` y detalle de celdas. Solo validación interna:
  advierte si >15% de IC excluyen 0; nunca se incorpora a tablas/figuras del paper.
- `diagnostics/closure_status.json`: completo/incompleto y revisión pendiente.

Los resultados reales (ICC, consistencia, placebo, cobertura) deben revisarse
después de ejecutar Colab. Los tests sintéticos no sustituyen esa corrida.
