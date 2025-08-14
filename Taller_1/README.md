# Análisis Exploratorio de Datos NHANES 2009-2012

## Descripción

Este proyecto presenta un análisis exploratorio exhaustivo de los datos de la Encuesta Nacional de Examen de Salud y Nutrición (NHANES) de Estados Unidos correspondientes al período 2009-2012. El análisis incluye exploración de datos, limpieza, transformaciones y desarrollo de modelos predictivos.

## Estructura del Proyecto

```
Taller_1/
├── NoteBook.ipynb          # Notebook principal con el análisis completo
├── Data/
│   └── NHANES2009-2012.csv # Dataset original
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## Contenido del Análisis

### 1. Instalación y Configuración
- Instalación de librerías necesarias
- Configuración del entorno de visualización

### 2. Carga y Descripción de los Datos
- Carga del dataset NHANES 2009-2012
- Diccionario completo de variables
- Descripción del contexto de la encuesta

### 3. Exploración y Análisis Exploratorio
- **Estructura del Dataset**: Análisis de tipos de variables
- **Calidad de los Datos**: Evaluación de valores nulos y duplicados
- **Estadísticas Descriptivas**: Análisis estadístico de variables numéricas
- **Análisis de Variables Numéricas**: Distribuciones, asimetría y curtosis
- **Análisis de Variables Categóricas**: Distribuciones y relaciones
- **Relaciones entre Variables**: Correlaciones y gráficos de dispersión

### 4. Preparación y Limpieza de los Datos
- **Eliminación de Variables Redundantes**: Reducción de dimensionalidad
- **Tratamiento de Valores Nulos**: Imputación estratégica
- **Detección de Datos Atípicos**: Algoritmos LOF e IQR

### 5. Transformación de Datos
- **Normalización**: Escalamiento Min-Max de variables numéricas
- **Codificación**: Transformación de variables categóricas
- **Variables Derivadas**: Creación de Income_Age

### 6. Modelado Predictivo
- **Predicción de BMI**: Modelos de regresión (Linear, Random Forest)
- **Clasificación de Diabetes**: Red neuronal feed-forward con balanceo de clases

### 7. Conclusiones y Hallazgos
- Síntesis de resultados
- Implicaciones para salud pública
- Limitaciones y direcciones futuras

## Instalación y Uso

### Prerrequisitos
- Python 3.8 o superior
- Jupyter Notebook o JupyterLab

### Instalación

1. **Clonar/Descargar el proyecto**
   ```bash
   # Si usa git
   git clone [URL_del_repositorio]
   cd Taller_1
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar Jupyter**
   ```bash
   jupyter notebook NoteBook.ipynb
   # o
   jupyter lab NoteBook.ipynb
   ```

### Uso del Notebook

El notebook está diseñado para ejecutarse de manera secuencial:

1. **Ejecute todas las celdas en orden** para reproducir el análisis completo
2. **Use los widgets interactivos** para explorar diferentes variables
3. **Modifique parámetros** según sus necesidades específicas

## Características Principales

### Widgets Interactivos
- Selección de variables para análisis de distribuciones
- Comparación interactiva de normalización
- Análisis de correlaciones dinámico
- Gráficos de dispersión multidimensionales

### Visualizaciones Avanzadas
- Gráficos de barras con Plotly
- Histogramas con estadísticas superpuestas
- Matrices de correlación interactivas
- Boxplots con detección de outliers
- Gráficos de dispersión con múltiples dimensiones

### Modelos Implementados
- **Regresión Linear**: Para predicción de BMI
- **Random Forest**: Modelo ensemble para BMI
- **Red Neuronal MLP**: Clasificación binaria de diabetes

## Dataset

**Fuente**: National Health and Nutrition Examination Survey (NHANES) 2009-2012
**Tamaño**: 10,000 observaciones, 75 variables
**Período**: 2009-2012
**Contenido**: Datos demográficos, socioeconómicos, antropométricos, de laboratorio y de salud

### Variables Principales
- **Demográficas**: Age, Gender, Race, Education, MaritalStatus
- **Socioeconómicas**: HHIncomeMid, Poverty, HomeRooms
- **Antropométricas**: Weight, Height, BMI
- **Salud Cardiovascular**: BPSysAve, BPDiaAve, Pulse
- **Laboratorio**: Cholesterol, Testosterone
- **Estilo de Vida**: PhysActive, SleepHrsNight, SmokeNow

## Resultados Esperados

Al ejecutar el notebook completo, obtendrá:

1. **Análisis descriptivo** completo del dataset
2. **Visualizaciones interactivas** para exploración de datos
3. **Dataset limpio y transformado** listo para modelado
4. **Modelos predictivos entrenados** con métricas de evaluación
5. **Insights** sobre patrones de salud en la población estadounidense

## Personalización

El código está estructurado con funciones reutilizables que permiten:

- **Cambiar parámetros** de modelos fácilmente
- **Aplicar a otros datasets** similares
- **Extender análisis** con nuevas variables
- **Modificar visualizaciones** según necesidades

## Contribución

Para contribuir al proyecto:

1. Fork el repositorio
2. Cree una rama para su feature (`git checkout -b feature/AmazingFeature`)
3. Commit sus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abra un Pull Request

## Licencia

Este proyecto está bajo la Licencia MIT - vea el archivo [LICENSE.md](LICENSE.md) para detalles.

## Contacto

**Grupo de Trabajo**: Anderson Alvarado, Maria Paula y Camila  
**Curso**: Gestión de Datos  
**Institución**: Pontificia Universidad Javeriana

## Reconocimientos

- **NHANES**: Por proporcionar los datos de salud pública
- **Scikit-learn**: Por las herramientas de machine learning
- **Plotly**: Por las capacidades de visualización interactiva
- **Jupyter**: Por el entorno de desarrollo interactivo