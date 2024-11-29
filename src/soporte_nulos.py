from itertools import product
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt


def outliers_isolation_forest(df, niveles_contaminacion, lista_estimadores):
    """
    Agrega columnas al DataFrame con la detección de outliers utilizando Isolation Forest
    para diferentes niveles de contaminación y números de estimadores.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada.
    - niveles_contaminacion (list): Lista de niveles de contaminación a probar (por ejemplo, [0.01, 0.05, 0.1]).
    - lista_estimadores (list): Lista de cantidades de estimadores a probar (por ejemplo, [10, 100, 200]).

    Returns:
    - pd.DataFrame: DataFrame con nuevas columnas para cada configuración de Isolation Forest.
    """
    # Seleccionar columnas numéricas
    col_numericas = df.select_dtypes(include=np.number).columns

    # Generar todas las combinaciones de niveles de contaminación y estimadores
    combinaciones = list(product(niveles_contaminacion, lista_estimadores))

    for cont, esti in combinaciones:
        # Inicializar Isolation Forest
        ifo = IsolationForest(
            n_estimators=esti,
            contamination=cont,
            n_jobs=-1  # Usar todos los núcleos disponibles
        )

        # Ajustar y predecir outliers
        df[f"outliers_ifo_{cont}_{esti}"] = ifo.fit_predict(df[col_numericas])
    
    return df


def visualizar_outliers(df, cols_numericas, figsize=(15, 10)):
    """
    Genera visualizaciones para las combinaciones de columnas numéricas utilizando las columnas
    que identifican outliers como `hue` en gráficos scatter.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos.
    - cols_numericas (list): Lista de columnas numéricas a combinar.
    - figsize (tuple): Tamaño de cada figura de visualización.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    # Crear todas las combinaciones de pares de columnas numéricas
    combinaciones_viz = list(combinations(cols_numericas, 2))
    columnas_hue = df.filter(like="outlier").columns

    for outlier in columnas_hue:
        # Calcular el número de filas y columnas necesarias para los subplots
        num_combinaciones = len(combinaciones_viz)
        ncols = min(3, num_combinaciones)  # Máximo 3 columnas por fila
        nrows = (num_combinaciones + ncols - 1) // ncols  # Calcular filas necesarias
        
        # Crear figura y ejes
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if num_combinaciones > 1 else [axes]  # Asegurar array plano
        
        for indice, (x, y) in enumerate(combinaciones_viz):
            sns.scatterplot(
                x=x,
                y=y,
                ax=axes[indice],
                data=df,
                hue=outlier,
                palette="Set1",
                style=outlier,
                alpha=0.4
            )
            axes[indice].set_title(f"{x} vs {y}")

        # Eliminar ejes vacíos si hay menos gráficos que subplots
        for ax in axes[len(combinaciones_viz):]:
            ax.axis("off")

        # Configuración general de la figura
        plt.suptitle(f"Outlier Analysis: {outlier}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar espacio sin solapar el título
        plt.show()