import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("NHANES2009-2012.csv")

# Inspeccionar variables numéricas
numericas = df.select_dtypes(include='number').columns.tolist()

# Declarar valores NA (según el diccionario)
df.replace({9999: pd.NA, 7777: pd.NA, 'Refused': pd.NA}, inplace=True)

# Estadísticas descriptivas
print(df[numericas].describe())

# Scatter: parejas sexuales vs edad, color por género
sns.scatterplot(data=df, x="SexNumPartYear", y="Age", hue="Gender")
plt.title("Número de parejas sexuales vs Edad")
plt.show()

# Scatter: ingreso vs edad diagnóstico de diabetes
sns.scatterplot(data=df, x="HHIncomeMid", y="DiabetesAge")
plt.title("Ingreso vs Edad al diagnóstico de Diabetes")
plt.show()

# Scatter: ingreso vs edad al primer hijo
sns.scatterplot(data=df, x="HHIncomeMid", y="Age1stBaby")
plt.title("Ingreso vs Edad al primer hijo")
plt.show()

# Gráfico avanzado ejemplo (3D Bubble plot con Plotly)
import plotly.express as px
fig = px.scatter(df, x="SmokeAge", y="AgeFirstMarij", color="HHIncomeMid",
                 size_max=10, title="Edad inicio fumar vs Marihuana, por ingreso")
fig.show()

