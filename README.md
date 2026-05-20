# CorVAE: Destilación de Datos Heterogéneos con Autoencoders Variacionales y selección de Coreset

Este repositorio contiene el código fuente de la arquitectura **CorVAE**, una metodología para la destilación de datos tabulares heterogéneos (con columnas numéricas y categóricas). CorVAE emplea Autoencoders Variacionales (VAEs) para aprender una representación latente de los datos, sobre la cual se aplican técnicas de selección de coreset para identificar un subconjunto de muestras representativas.

El objetivo principal es reducir significativamente el tamaño del conjunto de datos de entrenamiento sin sacrificar el rendimiento de los modelos de aprendizaje automático. Esto se traduce en una disminución de los costos computacionales, el tiempo de entrenamiento y el consumo energético, alineándose con las necesidades del Edge Computing. 

La metodología permite entrenar modelos de clasificación (tanto basados en redes neuronales como tradicionales) sobre este coreset, logrando un rendimiento comparable al obtenido con el conjunto de datos completo.

## Metodología

CorVAE se basa en un pipeline de dos etapas:

1.  **Aprendizaje de Representación**: Se entrena un Autoencoder Variacional (VAE) con una arquitectura basada en Transformers para capturar las dependencias complejas entre las características de los datos tabulares. El VAE aprende a comprimir los datos en un espacio latente de menor dimensionalidad.
2.  **Selección de Coreset**: Sobre el espacio latente generado por el VAE, se aplican diversas técnicas de coreset para seleccionar un subconjunto óptimo de muestras. Las técnicas implementadas incluyen:
    * K-Means
    * Agglomerative Clustering
    * Least Confidence
    * K-Center Greedy

Finalmente, el coreset seleccionado puede ser utilizado de dos maneras:
* **Entrenamiento en el Espacio Latente**: Los modelos se entrenan directamente con las representaciones latentes del coreset.
* **Entrenamiento en el Espacio Reconstruido**: Las muestras del coreset en el espacio latente se decodifican de vuelta al espacio original para entrenar los modelos.

## Estructura del Repositorio

```bash
├── data/
│   ├── adult/
│   ├── default/
│   └── shoppers/
├── results/
│   ├── checkpoint/
│   ├── curve_figures/
│   ├── effiency_figures/
│   ├── explainability_analysis/
│   └── exploratory_analysis/
├── tune_vae/
├── TDCOLER_data/
├── jobs/
├── Analysis_Datasets.ipynb
├── analisis_results.ipynb
├── Analysis_Results.ipynb
├── distill.py
├── environment.yml
├── main.py
├── models_vae.py
├── train_vae.py
└── utils.py
```

* **`main.py`**: Script principal que ejecuta el pipeline completo de destilación y evaluación.
* **`utils.py`**: Funciones para el preprocesamiento, división de datos, reconstrucción y evaluación de modelos con Optuna y validación cruzada.
* **`train_vae.py`**: Script para el entrenamiento del Autoencoder Variacional (VAE).
* **`distill.py`**: Implementación de los métodos de selección de coreset.
* **`models_vae.py`**: Definición de la arquitectura del VAE basada en Transformers.
* **`Analysis_Datasets.ipynb`**: Notebook con el Análisis Exploratorio de Datos (EDA) de los datasets utilizados (Adult, Default y Shopper).
* **`Analysis_Results.ipynb`**: Notebook para el análisis de resultados, eficiencia computacional, explicabilidad (T-SNE, coeficiente de silueta) y comparación con baselines.
* **`data/`**: Contiene los datasets y sus archivos de configuración `metadata.json`.
* **`results/`**: Almacena los resultados de las ejecuciones, incluyendo checkpoints, figuras de métricas y análisis.
* **`jobs/`**: Contiene scripts de ejecución (ej. `job_latent.sh`) y sus respectivos logs.

## Instalación

El entorno de desarrollo se gestiona con Conda. Para instalar las dependencias, clona el repositorio y ejecuta el siguiente comando:

```bash
conda env create -f environment.yml
conda activate corvae
```

La versión de Python utilizada es Python 3.12.

## Uso

### 1. Archivo de Configuración del Dataset

Para utilizar un nuevo conjunto de datos, es necesario crear un archivo `metadata.json` en una carpeta dentro de `data/`. Este archivo debe seguir el siguiente formato:

```json
{
"name": "adult",
"normalization": "z-score",
"num_col_idx": [0, 2, 4, 10, 11, 12],
"cat_col_idx": [1, 3, 5, 6, 7, 8, 9, 13],
"target_col_idx": [14],
"type": "classification",
"num_imputation": "mean",
"cat_imputation": "most_frequent",
"file": "dataset.csv",
"num_col_names": ["age", "fnlwgt", "education*num", "capital*gain", "capital*loss", "hours*per*week"],
"cat_col_names": ["workclass", "education", "marital*status", "occupation", "relationship", "race", "sex", "native*country"],
"target_col_name": "income"
}
```

### 2. Archivo de Hiperparámetros del VAE
Se requiere un archivo JSON con los mejores hiperparámetros para el VAE. Puedes usar los proporcionados en la carpeta `tune_vae/` como base.

```json
{
"best_params": {
"max_beta": 0.0077,
"min_beta": 0.00014,
"lambda_": 0.9,
"lr_pretrain": 0.0051,
"wd_pretrain": 0,
"d_token": 8,
"n_head": 2,
"factor": 16,
"num_layers": 1
}
}
```

### 3. Ejecución del Pipeline
Ejecuta el script `main.py` con los siguientes argumentos:

```bash
python3.12 main.py \
--metadata_path data/shoppers/metadata.json \
--distillation_space latent \
--epochs_latent 3000 \
--batch_size 4096 \
--checkpoint_path results/checkpoint/checkpoint_shoppers_Doriginal_F \
--hyperparams_vae tune_vae/tune_shoppers/best_hyperparams.json \
--kmeans_type centroid
```

`**distillation_space` Espacio donde se realiza la destilación. Opciones: `original` (no entrena el VAE) o `latent` (entrena el VAE y destila en el espacio latente).

## Salida de la Ejecución
La ejecución generará una carpeta de checkpoint (`results/checkpoint/checkpoint_shoppers_Doriginal_F` en el ejemplo) con la siguiente estructura:

`metrics.csv`: Fichero con los resultados de las métricas para cada modelo y semilla.

**Carpetas por técnica de destilación**: Contienen los mejores hiperparámetros encontrados para cada modelo.

**Carpeta vae**:

Espacios latentes (`Z_pred.pt`, `Y_label.pt`).

Redes `encoder.pt` y `decoder.pt`.

Carpeta `runs` con logs de TensorBoard para monitorizar la pérdida del VAE.

Carpeta `reconstructed_data` con los datos destilados y reconstruidos para cada IPC (Images Per Class) y semilla.

## Métricas y Evaluación
El rendimiento se evalúa utilizando un amplio rango de métricas de clasificación: `accuracy`, `precision`, `recall`, `f1-score`, `roc_auc` y `balanced_accuracy`. Debido al desbalance de clases presente en los datasets industriales, los resultados se reportan y analizan principalmente con **Balanced Accuracy**.

## Resultados
Los mejores resultados se observaron en conjuntos de datos heterogéneos y utilizando el **espacio reconstruido** (destilación latente seguida de reconstrucción con el decodificador del VAE). La combinación de **CorVAE + K-Means** demostró ser la más efectiva.
A continuación, se muestra un ejemplo de las curvas de rendimiento para el dataset Shopper, comparando el entrenamiento en el espacio latente (amarillo) y en el espacio reconstruido (azul).

![Conjunto shopper](results/curve_figures/shopper_KM_AG_balanced_accuracy.png)

## Baselines
Para validar la efectividad de CorVAE, los resultados se comparan contra un baseline consistente en el entrenamiento de los mismos modelos de clasificación utilizando el **conjunto de datos completo**. los modelos de selección de coreset en el espacio original y la arquitectura del estado del arte TDColER.

## Estudio de Ablación: Transformer vs. MLP

Para demostrar empíricamente que la arquitectura basada en Transformers es la que otorga la ventaja en datos heterogéneos, se implementó un estudio de ablación comparando **CorVAE (VAE + Transformer)** contra un **VAE estándar (solo capas densas/MLP)**.

### Arquitectura MLP-VAE

La variante MLP reutiliza el mismo `Tokenizer` y `Reconstructor` del CorVAE original, reemplazando únicamente los bloques Transformer del encoder y decoder por perceptrones multicapa. Esto garantiza una comparación justa donde la única variable es el backbone (Transformer vs. MLP).

Las clases relevantes son:
- `MLP_VAE_Core`: VAE con encoder/decoder MLP.
- `MLP_ModelVAE`: Modelo completo (MLP_VAE_Core + Reconstructor + ClassifierHead).
- `MLP_EncoderModel` / `MLP_DecoderModel`: Para inferencia standalone.

### Ejecución

**1. Seleccionar la arquitectura via CLI:**

```bash
python main.py \
    --metadata_path data/adult/metadata.json \
    --distillation_space latent \
    --epochs_latent 3000 \
    --batch_size 4096 \
    --checkpoint_path results/checkpoint/checkpoint_adult_MLP_Dlatent \
    --kmeans_type centroid \
    --epochs_fine_tuning_latent 0 \
    --hyperparams_vae tune_vae/tune_mlp_adult/best_hyperparams_mlp.json \
    --architecture mlp
```

**2. Buscar hiperparámetros del MLP-VAE con Optuna:**

```bash
cd tune_vae
python tune_mlp_vae.py --dataset adult --n_trials 30 --epochs 700
```

**3. Ejecutar el estudio completo para los 5 datasets:**

```bash
chmod +x run_ablation.sh
nohup bash run_ablation.sh > jobs/ablation_logs/ablation_full_log.txt 2>&1 &
```

El script `run_ablation.sh` ejecuta secuencialmente para cada dataset:
1. Búsqueda de hiperparámetros MLP con Optuna (si no existen).
2. Entrenamiento + destilación con `--architecture mlp`.

Los resultados se guardan en `results/checkpoint/checkpoint_{dataset}_MLP_Dlatent/metrics.csv`.

## Limitaciones
Actualmente, la metodología ha sido validada en:

Tareas de clasificación binaria.

Conjuntos de datos numéricos y heterogéneos (numéricos y categóricos).
La aplicación a conjuntos de datos puramente categóricos o a tareas de clasificación multiclase podría requerir adaptaciones en la arquitectura del VAE y en el pipeline de preprocesamiento.

## Reconocimiento
Este trabajo utiliza una versión adaptada de la arquitectura de **TabSyn**, originalmente diseñada para la generación de datos sintéticos con modelos de difusión.