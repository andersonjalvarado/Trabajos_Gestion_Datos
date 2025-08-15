# 📊 NHANES 2009-2012 Dashboard

Un dashboard profesional desarrollado en Streamlit para el análisis exploratorio y modelado predictivo de datos de salud y nutrición del National Health and Nutrition Examination Survey (NHANES) del período 2009-2012.

## 🚀 Características

### 📚 Diccionario de Datos
- **Información completa** del dataset NHANES 2009-2012
- **Categorización organizada** de variables por tipo:
  - Variables Demográficas
  - Variables Socioeconómicas  
  - Variables Físicas
  - Variables de Salud Cardiovascular
  - Variables de Estilo de Vida
  - Variables de Salud
  - Variables Reproductivas
- **Descripciones detalladas** y rangos de valores para cada variable

### 📊 Resumen y Métricas
- **Métricas clave** del dataset (participantes, variables, tipos)
- **Gráfico de distribución** de tipos de columnas
- **Análisis general** del DataFrame (nulos, duplicados)
- **Estadísticas descriptivas** completas de variables numéricas

### 🔍 Exploración de Datos
- **Visualización de valores nulos** por columna
- **Análisis de distribuciones** de variables numéricas
- **Detección de outliers** mediante boxplots
- **Análisis de variables categóricas** con distribuciones
- **Gráficos de relaciones** (pairplots y scatter plots multidimensionales)
- **Matriz de correlaciones** interactiva
- **Detección de outliers** usando algoritmos LOF e IQR

### ⚙️ Preparación y Transformación
- **Comparación de normalización** (original vs normalizado)
- **Codificación de variables categóricas** (Education, MaritalStatus)
- **Creación de variables derivadas** (Income_Age)
- **Tratamiento de outliers** (eliminar, reemplazar, ignorar)
- **Pipeline completo** de limpieza con función `clean_nhanes_data`

### 🤖 Modelado Predictivo
- **Predicción de BMI** (Regresión):
  - Linear Regression
  - Random Forest
  - Métricas: RMSE, MAE, R²
  - Importancia de características
  
- **Clasificación de Diabetes**:
  - Red Neuronal Multi-Layer Perceptron
  - Balanceamiento de clases con undersampling
  - Matriz de confusión interactiva
  - Reporte de clasificación completo

## 📋 Requisitos

```bash
streamlit==1.28.1
pandas==2.1.3
numpy==1.24.3
plotly==5.17.0
seaborn==0.12.2
matplotlib==3.7.2
scikit-learn==1.3.2
ipywidgets==8.1.1
```

## 🛠️ Instalación

1. **Clonar o descargar** el proyecto en la carpeta `Taller_1/Dashboard/`

2. **Instalar dependencias**:
   ```bash
   cd Dashboard
   pip install -r requirements.txt
   ```

3. **Verificar estructura de archivos**:
   ```
   Taller_1/
   ├── Dashboard/
   │   ├── dashboard.py
   │   ├── utils.py
   │   ├── requirements.txt
   │   └── README.md
   └── Data/
       └── NHANES2009-2012.csv
   ```

## 🚀 Uso

1. **Ejecutar el dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

2. **Acceder al dashboard**:
   - El dashboard se abrirá automáticamente en `http://localhost:8501`
   - Si no se abre automáticamente, copie la URL desde la terminal

3. **Navegación**:
   - Use la **barra lateral** para navegar entre secciones
   - Configure el **filtro de fechas** en el header (aplicable globalmente)
   - Interactúe con los **selectores y botones** en cada sección

## 📁 Estructura del Proyecto

### `dashboard.py`
- **Archivo principal** del dashboard Streamlit
- **Interfaz de usuario** y layout profesional
- **Integración** de todas las funcionalidades
- **Navegación** entre secciones
- **Styling CSS** personalizado

### `utils.py`
- **Funciones de análisis** extraídas del notebook
- **Funciones de visualización** con Plotly
- **Algoritmos de machine learning**
- **Pipeline de limpieza** de datos
- **Funciones de preprocesamiento**

### `requirements.txt`
- **Dependencias** necesarias del proyecto
- **Versiones específicas** para reproducibilidad

## 🎯 Funcionalidades Clave

### Filtrado Global
- **Filtro de fechas** en el header que se aplica a todas las visualizaciones
- **Configuración de período** 2009-2012

### Visualizaciones Interactivas
- **Todos los gráficos** utilizan Plotly para interactividad
- **Selección dinámica** de variables en widgets
- **Zoom, pan y hover** en todas las visualizaciones

### Análisis Profesional
- **Métricas con cards** estilizadas
- **Tablas informativas** con estadísticas
- **Progreso visual** con spinners y mensajes de estado

### Machine Learning
- **Modelos tradicionales** y redes neuronales
- **Evaluación completa** con múltiples métricas
- **Visualización de resultados** (matriz de confusión, importancia)

## 🔧 Configuración Avanzada

### Personalizar Styling
Modifique el CSS en `dashboard.py` en la sección:
```python
st.markdown("""
<style>
    .main-header { ... }
    .section-header { ... }
    .metric-card { ... }
</style>
""", unsafe_allow_html=True)
```

### Añadir Nuevos Análisis
1. **Crear funciones** en `utils.py`
2. **Integrar en dashboard.py** en la sección correspondiente
3. **Añadir navegación** en el sidebar si es necesario

### Modificar Pipeline de Limpieza
Edite la función `clean_nhanes_data()` en `utils.py` para:
- Cambiar criterios de eliminación de outliers
- Modificar estrategias de imputación
- Añadir nuevas transformaciones

## ⚠️ Notas Importantes

1. **Datos Originales vs Transformados**:
   - Las secciones de exploración usan datos originales
   - El modelado predictivo usa datos transformados del pipeline

2. **Rendimiento**:
   - Los datos se cargan y cachean automáticamente
   - Los modelos se entrenan bajo demanda

3. **Compatibilidad**:
   - Desarrollado y probado con las versiones especificadas
   - Compatible con Python 3.8+

## 🐛 Solución de Problemas

### Error: "No se pudo encontrar el archivo de datos"
- ✅ **SOLUCIONADO**: El dashboard ahora verifica múltiples rutas automáticamente
- El archivo debe estar en `../Data/NHANES2009-2012.csv` (carpeta hermana)
- Si persiste el error, verificar permisos de lectura del archivo

### Error de importación de dependencias
```bash
pip install --upgrade -r requirements.txt
```

### Dashboard no se abre
- Verificar que el puerto 8501 esté disponible
- Usar `streamlit run dashboard.py --server.port 8502` para otro puerto

## 📈 Extensiones Futuras

- **Más algoritmos** de machine learning
- **Análisis temporal** avanzado
- **Exportación de reportes** en PDF
- **Comparación de modelos** lado a lado
- **Análisis de sesgo** en los datos

## 👥 Contribución

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear rama para nueva funcionalidad
3. Implementar cambios con documentación
4. Crear pull request con descripción detallada

## 📄 Licencia

Este proyecto es parte del curso de Gestión de Datos y está destinado para fines educativos.

---

**Desarrollado para el análisis del dataset NHANES 2009-2012**  
*Dashboard profesional con Streamlit y análisis completo de datos de salud*