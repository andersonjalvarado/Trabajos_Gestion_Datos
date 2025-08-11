#!/usr/bin/env python3
"""
Análisis exploratorio del dataset NHANES 2009–2012

Este script guía al estudiante a través del proceso de:
1. Verificar librerías necesarias
2. Cargar y limpiar datos reales de salud pública
3. Explorar estadísticamente el contenido
4. Visualizar relaciones entre variables de forma gráfica

"""

# ────────────────────────────────────────────────────────────────
# SECCIÓN 1: VERIFICAR SI ESTÁN INSTALADAS LAS LIBRERÍAS NECESARIAS
# ────────────────────────────────────────────────────────────────

# Este bloque comprueba si las librerías necesarias están disponibles.
# Si alguna falta, se indica claramente en consola cómo instalarla.

required_modules = {
    "pandas": "pandas",            # Manipulación de datos en tablas
    "seaborn": "seaborn",          # Gráficos estadísticos simples
    "matplotlib": "matplotlib",    # Biblioteca base de gráficos
    "plotly.express": "plotly"     # Gráficos interactivos
}

missing = []

# Verifica cada módulo, y si no está disponible, lo agrega a la lista de faltantes
for name, import_name in required_modules.items():
    try:
        __import__(import_name)
    except ImportError:
        missing.append(name)

# Si falta alguna librería, se detiene el programa e informa cómo instalarla
if missing:
    print("\nFALTAN LIBRERÍAS NECESARIAS PARA EJECUTAR ESTE PROGRAMA:\n")
    for lib in missing:
        print(f"  - {lib}")
    print("\nPuedes instalarlas ejecutando este comando en tu terminal:\n")
    print(f"pip3 install {' '.join(missing)}\n")
    exit(1)

# ────────────────────────────────────────────────────────────────
# SECCIÓN 2: IMPORTAR LAS LIBRERÍAS
# ────────────────────────────────────────────────────────────────

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Configura el estilo gráfico por defecto para seaborn
# sns.set(style="darkgrid")    # fondo oscuro con grid
# sns.set(style="white")       # fondo blanco sin grid
# sns.set(style="dark")        # fondo oscuro sin grid
# sns.set(style="ticks")       # estilo minimalista con marcas en los ejes
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (11, 8)  # Tamaño estándar de gráficos en pulgadas

# ────────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR PARA PAUSAR ENTRE VISUALIZACIONES
# ────────────────────────────────────────────────────────────────

# Esta función muestra un mensaje y espera que el usuario presione ENTER
# antes de continuar. Ideal para clase o para ejecutar paso a paso.

def pausar(mensaje):
    input(f"\n{mensaje}\nPresiona [ENTER] para continuar...\n")

# ────────────────────────────────────────────────────────────────
# SECCIÓN 3: CARGA DEL ARCHIVO DE DATOS CSV
# ────────────────────────────────────────────────────────────────

print("Cargando el archivo de datos NHANES2009-2012.csv ...")

# Lee el archivo CSV con los datos de la encuesta NHANES
df = pd.read_csv("NHANES2009-2012.csv")

# Informa cuántas filas (personas) y columnas (variables) contiene el dataset
pausar(f"Archivo cargado correctamente: {df.shape[0]} filas, {df.shape[1]} columnas\n")

# ────────────────────────────────────────────────────────────────
# SECCIÓN 4: INSPECCIÓN Y LIMPIEZA DE LOS DATOS
# ────────────────────────────────────────────────────────────────

print("Identificando columnas numéricas...")

# Extraemos los nombres de las columnas que contienen datos numéricos
# Esto nos ayudará a enfocarnos en esas variables para análisis estadístico
num_cols = df.select_dtypes(include="number").columns
pausar(num_cols.tolist())

# LIMPIEZA DE VALORES FALTANTES O INVÁLIDOS

# Algunas columnas del dataset usan valores especiales para representar
# respuestas no válidas o faltantes, por ejemplo:
#   - 9999: "No responde" o "No aplica"
#   - 7777: "Desconocido"
#   - 'Refused': participante rechazó responder
# Los reemplazamos con pd.NA, que es el formato estándar de "faltante" en pandas

df.replace({9999: pd.NA, 7777: pd.NA, 'Refused': pd.NA}, inplace=True)

# MOSTRAR ESTADÍSTICAS BÁSICAS

print("\nEstadísticas descriptivas de las variables numéricas:\n")
# Describe muestra: media, desviación estándar, mínimos, máximos, etc.
pausar(df[num_cols].describe())

# ────────────────────────────────────────────────────────────────
# SECCIÓN 5: VISUALIZACIONES EXPLORATORIAS ESTÁTICAS CON SEABORN
# ────────────────────────────────────────────────────────────────

# Gráfico 1: Edad vs número de parejas sexuales en el último año

pausar("Vamos a presentar el gráfico: 'Parejas sexuales por año vs Edad'")

sns.scatterplot(data=df, x="SexNumPartYear", y="Age", hue="Gender")
plt.title("Parejas sexuales por año vs Edad")
plt.xlabel("Número de parejas sexuales en el último año")
plt.ylabel("Edad del participante")
plt.tight_layout()
plt.show()

# Gráfico 2: Ingreso medio del hogar vs Edad al diagnóstico de diabetes

pausar("Vamos a presentar el gráfico: 'Ingreso del hogar vs Edad al diagnóstico de diabetes'")

sns.scatterplot(data=df, x="HHIncomeMid", y="DiabetesAge")
plt.title("Ingreso del hogar vs Edad al diagnóstico de Diabetes")
plt.xlabel("Ingreso medio del hogar (USD)")
plt.ylabel("Edad al ser diagnosticado con diabetes")
plt.tight_layout()
plt.show()

# Gráfico 3: Ingreso vs edad al primer hijo

pausar("Vamos a presentar el gráfico: 'Ingreso del hogar vs Edad al primer hijo'")

sns.scatterplot(data=df, x="HHIncomeMid", y="Age1stBaby")
plt.title("Ingreso del hogar vs Edad al primer hijo")
plt.xlabel("Ingreso medio del hogar (USD)")
plt.ylabel("Edad al primer hijo")
plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────
# SECCIÓN 6: VISUALIZACIÓN INTERACTIVA CON PLOTLY
# ────────────────────────────────────────────────────────────────

# Este gráfico permite explorar visualmente tres dimensiones:
# Edad en que empezó a fumar, edad en que probó marihuana y nivel de ingresos

pausar("Vamos a presentar el gráfico interactivo: 'Edad de inicio al fumar vs Edad primer uso de marihuana'")

fig = px.scatter(
    df,
    x="SmokeAge",              # Edad al comenzar a fumar
    y="AgeFirstMarij",         # Edad al primer uso de marihuana
    color="HHIncomeMid",       # Color representa el ingreso medio del hogar
    title="Edad de inicio al fumar vs Edad primer uso de marihuana",
    labels={
        "SmokeAge": "Edad de inicio al fumar",
        "AgeFirstMarij": "Edad primer uso de marihuana",
        "HHIncomeMid": "Ingreso medio del hogar"
    }
)
fig.show()

# ────────────────────────────────────────────────────────────────
# SECCIÓN FINAL: MENSAJE DE CIERRE
# ────────────────────────────────────────────────────────────────

print("\nAnálisis finalizado.")
