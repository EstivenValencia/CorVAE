# Este repositorio contiene el codigo fuente de la arquitectura CorVAE
una metodología para la destilación de datos tabulares heterogeneos (columnas numéricas y categoricas) que aplica autoencoders variacionales VAEs 
sobre cuyo espacio latente se aplican técnicas de coreset como K-Means, Agglomerative clustering, Least confidence y K-Center greedy para seleccionar un conjunto de muestras representativas (coreset) que permite entrenar modelos de clasificación basados en redes neuronales como aquellos que no y que alcancen 
rendimientos comparables al entrenarse con todos los datos

# Métricas utilizadas y modelos evaluadas
El presente proyecto aunque calcula la mayoría de métricas como accuracy, recall, presicion,
f1-score, roc auc y balanced accuracy los resultados que se reportan se analizaron principalmente con 
balanced accuracy debido al desbalance de clases de los modelos

# Resultados
En la siguiente grafica se presentan las curvas de la métrica de balanced accurcy para el conjunto de 
shopper destilado desde un IPC de 10 hasta 100, a nivel de fila se encuentra la técnica de coreset y a 
nivel de columna el modelo, las graficas sombreadas en azul son la destilación en latente y reconstrucción con el decodificador del VAE para entrenar los modelos, el amarrillo es el entrenammiento 
permaneciendo en el espacio latente. 

Los mejores resultados se evidenciaron en conjuntos de datos heterogeneos y en el espacio reconstruido
<Poner aqui la imagen de shopper>

# Version de python
La version de python utilizada es python 3.12

# Reconocimiento
Este trabajo utiliza una version adaptada de la arquitectura de tabsyn utilizada para 
la generación de datos con modelos de difusión.

# Baseline
Para comparar CorVAE nos comparamos contra el entrenamiento de 

# Estructura 
main.py script principal que ejecuta el pipeline completo con todos los métodos de destilación y el 
entrenamiento con los datos completos. Para los métodos de destilación se pueden aplicar en el espacio original, en el espacio latente reconstruyendo al espacio original o permaneciendo al espacio latente

utils.py contiene todas las funciones para el preprocesado de lso datos, la división, la recontrucción de los datos y la evaluación de los modelos con optuna durante 3o trials con un cross validation de 5 folds

train_vae.py Este script contiene las funciones para el entrenamiento completo de la parte del VAE y almacena en un tesorboard las diferentes perdidas durante el entrenamiento

distill.py Este script contiene la definición de los métodos de selección de coreset utilizados para la 
destilación de los datos

models_vae.py Contiene la definición de los modelos transformers de las redes del VAE del tokenizador y todas las funciones que necesita el script train_vae para funcionar

Analisis_datasets.ipynb este notebook contiene el análisis exploratorio de los tres conjuntos de datos que se trabajaron en este proyecto (Adult, Default y shopper)

Analisis_results.ipynb este notebook contiene el análisis de todos los resultados que se obtuvieron con CorVAE para los diferentes conjuntos y modelos, así como también los analisis de eficiencia computacional, tamaño de los datos destilados y analisis de explicabilidad del funcionamiento del latente utilizando t-sne y coeficiente de silueta. Además presenta la comparación contra los baselines

results En la carpeta contiene todas las figuras de los analisis de resultados así como tambien los checkpoints con las corridas para los diferentes conjuntos de datos

tune_vae Esta carpeta contiene las primeras corridas que se utilizaron para tunear los hiperparametros de la red del VAE

data Esta carpeta contiene los archivos de configuracion para los diferentes conjuntos asi como sus archivos de configuracion

TDCOLER_data Esta carpeta contiene los resultados de la arquitectura TDColER para los conjutnos de Adult y default especificamente para el autoencoder con capas transformers sin capa clasificadora 

job.sh presenta un ejemplo de ejecución del archivo main.py

# Instalación de dependencias
El codigo se ejecuto utilizando un entorno de conda, para su instalación debe ejecutar
conda create conda env create -f environment.yml

# Ejecución de un nuevo conjunto
Para la ejecución de un nuevo conjunto de datos, se requiere un archivo de configuración json con la 
información necesaria para ejecutar el pipeline, en data se encuentra los ejemplos del conjunto de Adult, Default y Shopper, este debe tener el siguiente formato: 

{ 
    "name": "adult", # Nombre del conjunto de datos
    "normalization": "z-score", # La normalización que se aplica a las columna numéricas puede ser
    z-score, min-max y robust
    "num_col_idx": [0, 2, 4, 10, 11, 12], # Indices de las columnas númericas en el dataset
    "cat_col_idx": [1, 3, 5, 6, 7, 8, 9, 13], # Indices de las columnas categóricas en el dataset
    "target_col_idx": [14], #Indice de la columna con la etiqueta
    "type": "classification", # Tarea para la que se va utilizar el dataset (por ahora solo clasificación)
    "num_imputation": "mean", # Método de imputación para las columnas númericas tiene las opciones de mean, median y most_frequent
    "cat_imputation": "most_frequent", # Método de imputación para las columnas categóricas solo most_frequent
    "file": "dataset.csv", # Nombre del archivo csv con los datos debe estar al mismo nivel que el archivo de configuración
    "num_col_names": ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"], # Nombres de las columnas numéricas
    "cat_col_names": ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"], # Nombres de las columnas categoricas
    "target_col_name": "income" # Nombre de la columna objetivo
}

a partir de definir el archivo de hiperparametros se puede ejecutar el script main.py de la siguiente manera: 

main.py

# Limitaciones
Por ahora solo se ha probado la metodoloǵia en tareas de clasificación y en conjuntos con datos categoricos y numéricos y solo numéricos, aplicarlo a conjntos solo categóricos requiere cambios en la estructura. 

# Resultados
La mejor combinación se obtuvo para la mezcla de K-means con CorVAE

 que aplica autoencoders variacionales VAEs para mejorar 
las capacidades de los métodos de coreset como K-MEANS y agglomerative clustering para la