# Proyecto 8: Predicción de retención de empleados

![imagen](images/header.jpg)


## Planteamiento del problema 🏢 🏢**

- Este proyecto forma parte de un máster de formación en Data Science e Inteligencia Artificial.

- Esta vez, nos toca trabajar en Recursos Humanos, y enfrentarnos a uno de los mayores dolores de cabeza de cualquier empresa: la rotación de empleados. ¿Por qué algunas personas deciden quedarse mientras otras se van? ¿Será el salario? ¿Las horas extra? ¿La relación con su jefe?

- En este proyecto, usaremos datos recopilados de una empresa ficticia (¡no, no es información confidencial!) que incluye desde encuestas de satisfacción hasta métricas de desempeño y horarios laborales. La tarea consiste en desentrañar patrones, analizar tendencias y construir un modelo que pueda predecir si un empleado permanecerá o decidirá decir adiós.

- Pero esto no es solo sobre números y gráficos; se trata de entender cómo las decisiones empresariales impactan la vida de las personas y cómo, con un poco de análisis, podríamos ayudar a las empresas a ser mejores lugares para trabajar. Así, nos preparamos para explorar datos, ensuciarnos las manos con algoritmos y, quién sabe, tal vez descubrir el secreto para mantener a los empleados felices y comprometidos.

- Este proyecto va más allá de un ejercicio técnico, también es un entrenamiento toma de decisiones basadas en datos. 


## Objetivos del Proyecto

- El principal desafío de este proyecto es abordar una de las preguntas más importantes para cualquier departamento de Recursos Humanos: ¿qué empleados tienen más probabilidades de quedarse en la empresa y cuáles podrían decidir irse? Para lograr esto, el trabajo será construir un modelo de machine learning capaz de predecir si un empleado permanecerá en la empresa o decidirá marcharse. 

- El enfoque no solo será técnico. A través del análisis de los datos, identificaremos cuáles son los factores más influyentes en la retención o rotación del personal. Por ejemplo:
   - ¿Es la satisfacción laboral un predictor clave?
   - ¿Tienen más probabilidades de irse quienes trabajan largas horas o aquellos con relaciones tensas con sus jefes?
   - ¿Qué papel juegan las promociones o los aumentos de salario?

- El modelo debe ser capaz de responder estas preguntas y ofrecer predicciones precisas que puedan usarse para tomar decisiones informadas. Esto significa que, además de construir un modelo que funcione, se deben  interpretar sus resultados y proponer estrategias basadas en ellos.


## Estructura del repositorio

El proyecto está construido de la siguiente manera:

- **datos/**: Carpeta que contiene archivos `.csv`, `.json` o `.pkl` generados durante la captura y tratamiento de los datos.

- **images/**: Carpeta que contiene archivos de imagen generados durante la ejecución del código o de fuentes externas.

- **notebooks/**: Carpeta que contiene los archivos `.ipynb` utilizados en la captura y tratamiento de los datos. Están numerados para su ejecución secuencial, y contenidos dentro de 5 carpetas, una para cada modelo, conteniendo cada una de ellas:
  - `1_EDA`
  - `2_Encoding`
  - `3_Outliers`
  - `4_Estandarización`
  - `5_Modelos`
  - Únicamente en el modelo 5: `6_Predicciones`

- **src/**: Carpeta que contiene los archivos `.py`, con las funciones y variables utilizadas en los distintos notebooks.

- `.gitignore`: Archivo que contiene los archivos y extensiones que no se subirán a nuestro repositorio, como los archivos .env, que contienen contraseñas.


## Lenguaje, librerías y temporalidad
- El proyecto fué elaborado con Python 3.9 y múltiples librerías de soporte:

*Librerías para el tratamiento de datos*
- [Pandas](https://pandas.pydata.org/docs/)
- [Numpy](https://numpy.org/doc/)

*Librerías para captura de datos*
- [Selenium](https://selenium-python.readthedocs.io)
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Requests](https://pypi.org/project/requests/)

*Librerías para gestión de tiempos*
- [Time](https://docs.python.org/3/library/time.html)
- [tqdm](https://numpy.org/doc/)

*Librerías para gráficas*
- [Plotly](https://plotly.com/python/)
- [Seaborn](https://seaborn.pydata.org)
- [Matplotlib](https://matplotlib.org/stable/index.html)
- [shap](https://shap.readthedocs.io/en/latest/)

*Librería para controlar parámetros del sistema*
- [Sys](https://docs.python.org/3/library/sys.html)

*Librería para controlar ficheros*
- [os](https://docs.python.org/3/library/os.html)

*Librería para generar aplicaciones basadas en Python*
- [streamlit](https://docs.streamlit.io)

*Librería para generar APIs basadas en Python*
- [flask](https://flask.palletsprojects.com/en/stable/)

*Librería para creación de modelos de Machine Learning*
- [scikitlearn](https://scikit-learn.org/stable/)

*Librería para la gestión del desbalanceo*
- [imblearn](https://imbalanced-learn.org/stable/)

*Librería para creación de iteradores (utilizada para combinaciones)*
- [itertools](https://docs.python.org/3/library/itertools.html)

*Librería para la gestión de avisos*
- [warnings](https://docs.python.org/3/library/warnings.html)


- Este proyecto es funcional a fecha 1 de diciembre de 2024.


## Instalación

1. Clona el repositorio
   ```sh
   git clone https://github.com/davfranco1/Proyecto8-PrediccionRetencionEmpleados.git
   ```

2. Instala las librerías que aparecen en el apartado anterior. Utiliza en tu notebook de Jupyter:
   ```sh
   pip install nombre_librería
   ```

3. Cambia la URL del repositorio remoto para evitar cambios al original.
   ```sh
   git remote set-url origin usuario_github/nombre_repositorio
   git remote -v # Confirma los cambios
   ```

4. Ejecuta el código en los notebooks, modificándolo si es necesario.

5. Para utilizar la app de Streamlit, copia el repositorio, abre una terminal en la carpeta base, y ejecuta el comando `streamlit run main.py`, que abrirá un navegador donde se ejecuta automáticamente el código. Recuerda que antes debes haber instalado la librería `pip install streamlit`.


## Conclusiones y próximos pasos

- XX


## Autor

David Franco - [LinkedIn](https://linkedin.com/in/franco-david)

Enlace del proyecto: [https://github.com/davfranco1/Proyecto8-PrediccionRetencionEmpleados](https://github.com/davfranco1/Proyecto8-PrediccionRetencionEmpleados)
