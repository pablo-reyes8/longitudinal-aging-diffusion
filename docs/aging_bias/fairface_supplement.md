# Suplemento transversal opcional de FairFace

En `notebooks/Aging_bias_references.ipynb`, configura únicamente
`FAIRFACE_ROOT = Path("/content/drive/MyDrive/FairFace_Frontal")`.
El valor predeterminado `None` conserva el comportamiento anterior.
`ReferenceConfig.fairface_root` se hereda al crear la configuración DINO.

La carpeta debe contener `age_predictions.csv` (`filename,age_pred`) y las
fotografías, directamente o en `images/`. Si existe
`metadata_frontal.csv` (o su JSON), se utiliza como lista permitida;
en su ausencia se utiliza `metadata.csv`, si existe. Solo se incorporan fotos
presentes con edad numérica finita entre 0 y 120. Si hay etiquetas demográficas,
solo se admite Latino/Hispanic; si hay columna `split`, solo `train`.
Las rutas antiguas de metadata se ignoran. Duplicados con edades contradictorias,
nombres ambiguos y suplementos vacíos se rechazan antes de cargar redes.

Todas las fotos pasan por el preprocesamiento y controles de calidad habituales.
Cada foto externa recibe una identidad única `fairface::<filename>`:
contribuye a media/desviación de normalización, curva transversal colombiana
ampliada y candidatos para PCA DINO, pero no crea información longitudinal ni
pares test-retest. El muestreo PCA sigue limitado por `embedding_fit_max_images`;
no implica que todas las fotos entren al ajuste PCA. Las curvas longitudinales
en unidades estandarizadas pueden cambiar al cambiar la escala del conjunto.

Los parquet conservan `population_source` y `age_source`; la configuración
guardada incluye la ruta opcional. Esto funciona también al recalcular las
sensibilidades y referencias DINO desde los parquet en el módulo 2.

Las edades son **predicciones**, no anotaciones cronológicas. Latino/Hispanic
es una etiqueta del dataset, no una nacionalidad ni una población colombiana
representativa. El suplemento no garantiza mejorar ICC, identificación o sesgo.
Debe declararse la procedencia del estimador de edades; si coincide con el
evaluador, no es evidencia de calibración cronológica independiente. Conserva
una corrida sin suplemento para sensibilidad. Tras cambiar las referencias,
recalcula el análisis del módulo 2, sin regenerar imágenes ni repetir picking.

DINO utiliza `device="auto"`: GPU si CUDA está disponible, encoder residente
durante extracción, sin transferencias por imagen; PCA y vectores guardados
permanecen en CPU. No se descarga ni ejecuta nada al importar estos módulos.
