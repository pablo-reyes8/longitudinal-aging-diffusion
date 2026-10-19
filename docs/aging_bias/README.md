# Evaluación comparativa de sesgo v2

## Orden de ejecución (Colab)

1. `notebooks/Aging_bias_references.ipynb`: preparar FaceLandmarker, declarar
   train y exclusiones de test, ajustar referencias, revisar ICC, soporte y
   descriptivo poblacional. Requiere imágenes FFHQ-Aging y su CSV oficial;
   `data/sample_kaggle` **no** es ese proxy. No descarga todo un dataset.
2. `notebooks/Paper_comparison_bias_v2.ipynb`: preparar CPU AdaFace + FairFace
   (+ DEX solo apéndice, KID), apuntar a los PNG existentes de la comparación,
   recalcular sobre el panel común y revisar las anclas reales.
3. Solo si las anclas pasan y fueron revisadas: marcar
   `ANCHORS_REVIEWED=True` y ejecutar la sección SD1.5 antes/después. Esta sí
   genera imágenes; no es parte de la importación ni del recálculo de métricas.
4. Evaluar las imágenes SD1.5 con el mismo subconjunto aging para todos los
   modelos. Generar curvas por strength y comparación descriptiva a D común.

La función principal nueva es
`evaluate_paper_v2(PaperV2Config(...), bundle=evaluation_bundle)`.
No usa el bundle de difusión como evaluador. Para nuevas generaciones de
literatura se mantiene `load_aging_models(..., load_metrics=False)` y sus
sesiones residentes existentes. No se cambia ninguna firma legacy.

## Archivos

- `per_sample.parquet` y copia CSV: atributos estandarizados, componentes de
  desplazamiento, pesos del panel común, métricas operativas, hashes y banderas.
- `bias_indices.csv`: común + completo por modelo, esquemas de ponderación,
  all/aging/rejuvenation; coeficientes, identificación, índices e IC por ID.
- `bias_indices_by_bin.csv`: el mismo análisis condicionado a horizonte.
- `paired_differences.csv`: diferencias ours − baseline en réplicas compartidas.
- `operational_metrics.csv`: delta_id_gain, identidad, edad independiente
  relativa y DEX de apéndice. El oráculo se separa en `validation/`.
- `kid_metrics.csv`: KID **sin ponderar**, sobre el panel común; no se etiqueta
  engañosamente un KID común como bin-balanced u operational.
- `tables/`: tablas 1, 2 y A1 en CSV/LaTeX; `pi_by_delta_bin.png`.
- `validation/`: self-checks, anclas reales y resultados del oráculo. El oráculo
  no aparece como competidor en las tablas del paper.
- `cache/`: atributos y evaluadores por SHA-256 + backend/configuración.
- `run_metadata.json`: configuraciones, hashes, versiones, commit, panel y
  fusiones. Los merges específicos de cada análisis están en bias_indices.
- SD1.5: `generated/{modelo}/{id}/{sample_id}.png` + JSON, manifest,
  failures y configuración. Las imágenes de protocolos anteriores se mueven
  recuperablemente a `stale/` si es necesario regenerarlas; nunca se evalúan
  como si pertenecieran al nuevo protocolo tras una regeneración fallida.

## Interpretación y limitaciones

- D, pi, drift y homog son estimandos del modelo estructural propuesto, no
  probabilidades étnicas ni medidas de nacionalidad. No se restringen a [0,1].
- `no_edit` evita confundir una copia de la fuente con mitigación de sesgo.
  `not_identified` invalida razones, no autoriza elegir otras para favorecer un
  modelo. Si las curvas poblacionales son colineales no hay atribución posible.
- La invarianza al strength es una predicción del modelo teórico, no una
  propiedad que se impone a los datos reales. Con w heterogéneo, la regresión
  agrupada no es necesariamente un promedio convexo de los w por horizonte;
  por eso se informa el análisis por bin además del agregado.
- Las CI estructurales son bootstrap por identidad **condicionales a las
  referencias ajustadas**. El descriptivo estima separadamente incertidumbre
  de las curvas C/P. No son CI conjuntas de toda la estimación de referencias.
- Sin repetidas cercanas confiables no hay atributos activos. No se degrada a
  ICC=1. La geometría depende de landmarks aprendidos; las regiones clásicas
  de piel/cabello son heurísticas, no segmentaciones perfectas.
- FFHQ es un proxy de dominio, no representa exactamente LAION ni toda una
  población europea/estadounidense. Colombia tampoco es una apariencia única.
- Error relativo FairFace no garantiza cancelar sesgo. Se usa la cabeza de
  edad del ResNet34 oficial, sin clasificar etnia; midpoint del grupo 70+ = 85.
  El crop alineado AdaFace difiere del crop dlib del script FairFace original:
  hay que revisar comportamiento en targets reales. DEX/MiVOLO se rechazan
  como evaluadores independientes. Procedencia no verificada permanece unknown.
- El SD1.5 contrafactual usa el mismo protocolo directo/CFG del proyecto; no
  pretende ser idéntico al pipeline img2img stock de Diffusers. Se restaura la
  conv_in **preentrenada**, incluidos los cuatro canales/bias que fueron
  ajustados durante entrenamiento. Restaurar solo DoRA no sería vanilla.
- La interpolación a D emparejado es descriptiva, solo dentro del soporte;
  no tiene IC de interpolación. Las curvas originales sí tienen IC por ID.
- DINOv2/PCA es opcional: ajustar en crops comunes de train+proxy, guardar PCA,
  construir otras referencias y proporcionar `robustness_bundle` y
  `robustness_reference`. Nunca se ajusta PCA con test. Retrieval de galería
  es opcional en la guía y no forma parte de esta primera implementación.

## Pruebas

Pruebas rápidas, sin modelos ni red:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest \
  tests/aging_bias tests/comparisong_v2 tests/legacy -q
```

La prueba T11 de cobertura Monte Carlo está marcada slow y solo se ejecuta
explícitamente con `AGING_BIAS_SLOW=1`. No forma parte de la verificación rápida.
Las pruebas de notebooks solo analizan/compilan celdas; no ejecutan descargas.

La suite legacy inicial tenía 380 aprobadas y dos fallos previos: el notebook
de ablaciones usa device cuda donde su test espera cpu; y un test del runner
inyecta un metrics_bundle incompleto. No se alteran esos módulos para ocultarlos.
El test del antiguo notebook se actualiza porque el usuario pidió quitarle la
comparación cuantitativa y dejarlo exclusivamente como generador de sweeps.
