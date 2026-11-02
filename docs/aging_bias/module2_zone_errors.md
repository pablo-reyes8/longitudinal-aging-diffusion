# Módulo 2: error longitudinal observado por zona

El módulo 1 cerrado se consume como **solo lectura**. No se cambian sus
atributos, ICC, signos, pesos, normalización, percentil IOD ni PCA. Las firmas de
`evaluate_paper_v2`, `run_paper_analysis` y `run_sd15_before_after` se conservan.
Un bundle con `metadata.module1_close=True` activa el protocolo nuevo. Los
bundles anteriores conservan el protocolo legacy de descomposición.

## Estimando

Para medida j, se usa la coordenada train-estandarizada z del módulo 1 y su
signo s fijado en train. Las zonas ya contienen su orientación interna.

```
model_change = s * (z_generated - z_source)
real_change  = s * (z_target    - z_source)
error        = model_change - real_change = s * (z_generated - z_target)
```

El CSV conserva ambos cambios, error firmado, error absoluto, signo y error
nativo sin orientación adicional. Positivo es más envejecido que la foto real
destino; negativo menos envejecido. No implica que el modelo no edite: el
baseline `source_noedit` tiene cambio cero y error −cambio real. El oráculo
`oracle_target` debe tener error exactamente cero.

Las fotos reales incluyen pose, expresión, iluminación, resolución y época de
captura, no únicamente envejecimiento biológico. Los controles del módulo 1
mitigan estas diferencias pero no las eliminan. Esta medida por sí sola no
identifica causalmente sesgo étnico/nacional ni confirma estereotipos.

## Agrupación y comparación

Se reportan los índices de zona activos y sus componentes activos, sin recalcular
ICC en test ni inventar zonas inactivas. Frente/surco/mandíbula se inspeccionan
en `ffhq_pattern_aging.csv`: signos esperados nativos −/−/+, respectivamente.
Se reportan IC e indicadores descriptivos por componente, no una prueba conjunta
ni una puntuación causal de sesgo. `ffhq_sign_agreement` compara descriptivamente
el error con la diferencia poblacional transversal FFHQ−latam congelada. Se
omite esa comparación donde las curvas extrapolan, pero no se elimina por ello
un error observable frente a una foto destino real.

El panel primario es común entre todos los modelos solicitados **por medida**.
Una zona ausente no invalida automáticamente otras zonas; no se imputan ceros.
Los paneles completos por modelo quedan como diagnóstico, no como ranking
directamente comparable. Cada fila incluye n de pares e identidades.

Se separan all/aging/rejuvenation y long_aging/long_rejuvenation (|delta|≥20).
El subconjunto largo nunca se mezcla con corto para ocultar bajo soporte.
Dentro de cada subconjunto se usan los pesos establecidos (`WeightingConfig`);
los merges internos de bins quedan registrados. El reporte principal usa el
soporte de edad test-retest del módulo 1. Todas las edades quedan también como
`all_ages_exploratory`; ICC adulto no valida niños.

Los IC usan bootstrap de identidades, con el mismo draw para todos los modelos
del panel. Recalculan pesos/bins en cada réplica. La implementación comprimida
se verifica contra draws con identidades duplicadas explícitas. Los IC están
condicionados a las medidas, máscaras y parámetros congelados; no propagan toda
la incertidumbre del ajuste del módulo 1. Con una identidad, sin bootstrap o
demasiados draws inválidos, los IC no se presentan como confiables.

## DINO y referencias congeladas

`run_paper_analysis` lee `REFERENCE_DIR/dinov2/embedding_pca.npz` y su bundle,
verifica hash y carga el encoder en GPU si hay CUDA, una vez por fase. No
escanea train/FFHQ, no reajusta PCA y no escribe dentro del módulo 1. Si falta
ese artefacto, falla antes del análisis largo. Las coordenadas PCA se reportan
como robustez latente; **no se asignan a zonas anatómicas** y sus signos no
tienen interpretación física de envejecimiento.

El pipeline nuevo no depende de D≈1 ni de una mezcla π identificada: un error
observado frente a GT no requiere esa regresión poblacional. Las validaciones
nuevas se identifican por `protocol=observed-longitudinal-zone-error-v1` y no
certifican las ratios anteriores. `bias_indices` queda vacío en este protocolo;
`summary` contiene resultados por zona, manteniendo las claves públicas útiles.
Edad independiente/AdaFace se guardan como métricas secundarias. DEX/KID siguen
siendo opcionales; el notebook no los solicita para evitar critic contaminado,
dependencias y cómputo que no intervienen en el error por zona.

## SD1.5 antes/después

Un bloque independiente ejecuta el mismo `run_sd15_before_after(sd_config)`:
pares de envejecimiento, misma semilla, strength, pasos y opciones de inferencia.
Vanilla restaura los cuatro canales originales y bias de conv_in, anula canales
de fuente y desactiva adaptadores/condicionadores. Después aplica el sistema
adaptado completo. Es el backbone SD1.5 bajo el protocolo img2img del proyecto,
no la pipeline stock de Diffusers ni un experimento que aísle exclusivamente
los datos de fine-tuning frente a cambios arquitectónicos.

El preflight comprueba extracción real de zonas, artefactos y compatibilidad del
DINO congelado antes de generar. No garantiza éxito de todas las imágenes
futuras ni del checkpoint de difusión. El bloque se ubica antes de literatura
para que un fallo de generación no desperdicie evaluación previa de otros
modelos. `resume=True` reutiliza PNG/JSON compatibles; fallos por transición se
registran. Los modelos no se descargan/offload por imagen.

`sd15_paired_zone_errors.csv` compara después−antes **a strength idéntico**,
con IC compartidos. Una diferencia MAE negativa favorece después; no se elige
a posteriori el strength que mejor conviene. Los fallos se documentan y la
comparación usa un panel común; revisar cobertura para sesgo de selección.

## Notebook y pares largos

`notebooks/Paper_comparison_bias_v2.ipynb` tiene 9 celdas. Usa las rutas del
cuaderno proporcionado en `output/comparsing_metrics/v1_metricas_mal/`, consume
las generaciones de literatura y picking existentes, y guarda todo lo nuevo en
`paper_comparison_zones_v2`. No ejecuta picking ni sobrescribe el módulo 1.
Las tablas principales y las figuras se muestran automáticamente.

`diagnostics/long_pair_support.csv` informa soporte real sin fusionar horizontes.
`long_pair_candidates.csv` inventaría fotos del mismo test root, restringidas a
las identidades del manifest actual. Lista combinaciones largas que faltan, sin
modificar el manifest ni generar imágenes. Para incorporarlas hay que generar
TODOS los modelos sobre ellas y versionar la evaluación; no se pueden inventar
nuevos pares longitudinales reales. El test actual no cambia.

La implementación se verifica con tests sintéticos CPU. Las métricas y figuras
reales requieren ejecutar el notebook; no se descargaron pesos ni se ejecutó
difusión durante la implementación.
