import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample

def plot_distributions_cant(df):
    """
    Función que recibe un dataframe y muestra un gráfico de barras con la cantidad
    de columnas numéricas y categóricas.
    
    Parámetros:
    df (DataFrame): El dataframe a analizar
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    counts = {'Numéricas': len(num_cols), 'Categóricas': len(cat_cols)}
    
    colors = ['royalblue', '#EF553B']
    fig = px.bar(x=list(counts.keys()), y=list(counts.values()),
                 labels={'x': 'Tipo de columna', 'y': 'Cantidad'},
                 title='Cantidad de columnas numéricas y categóricas',
                 text=list(counts.values()))
    fig.update_traces(marker_color=colors)
    fig.update_layout(title_x=0.5, title_font_size=20, 
                      xaxis_title_font_size=16, yaxis_title_font_size=16)
    fig.update_layout(xaxis_tickfont_size=14, yaxis_tickfont_size=14)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(bargap=0.2, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_yaxes(range=[0, max(counts.values()) + 1])
    fig.update_xaxes(tickangle=0, tickmode='array', tickvals=list(counts.keys()),
                     ticktext=list(counts.keys()))
    fig.update_layout(showlegend=False)
    fig.update_layout(font=dict(family='Arial, sans-serif', size=14, color='black'))
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, showline=False),
                      yaxis=dict(showgrid=False, zeroline=False, showline=False))
    fig.update_layout(title_font_family='Arial, sans-serif', title_font_color='black')
    fig.update_layout(xaxis_title='Tipo de columna', yaxis_title='Cantidad')
    fig.update_layout(legend_title_text='Tipo de columna')          
    
    return fig

def analize_dataframe(df):
    """
    Función que recibe un dataframe y retorna información sobre su estructura,
    valores nulos y duplicados en formato de DataFrames.
    
    Parámetros:
    df (DataFrame): El dataframe a analizar
    """
    # Nulos por columna
    nulos_col = df.isnull().sum().to_frame('Nulos')
    nulos_col['% Nulos'] = (df.isnull().mean() * 100).round(2)
    
    # Total de nulos
    total_nulos = df.isna().sum().values.sum()
    
    # Duplicados
    n_duplicados = len(df) - len(df.drop_duplicates())
    pct_duplicados = 100 * n_duplicados / len(df)
    duplicados_df = pd.DataFrame({'Duplicados': [n_duplicados], '% Duplicados': [pct_duplicados]})
    
    return {
        'shape': df.shape,
        'nulos_por_columna': nulos_col,
        'total_nulos': total_nulos,
        'duplicados': duplicados_df
    }

def basic_statistics(df):
    """
    Calcula estadísticas básicas para las variables numéricas en el DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        
    Returns:
        pandas.DataFrame: DataFrame con las estadísticas calculadas.
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    stats = df[numerical_cols].describe().T
    stats['skew'] = df[numerical_cols].skew()
    stats['kurtosis'] = df[numerical_cols].kurtosis()
    stats['missing'] = df[numerical_cols].isnull().sum()
    stats['% missing'] = (df[numerical_cols].isnull().mean() * 100).round(2)
    
    return stats.T

def plot_nulls(df):
    """
    Función que recibe un dataframe y muestra un gráfico de barras con la cantidad
    de valores nulos por columna.
    
    Parámetros:
    df (DataFrame): El dataframe a analizar
    """
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    
    if len(null_counts) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No hay valores nulos en el dataset",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title='Valores Nulos por Columna')
        return fig
    
    fig = px.bar(null_counts, 
                 labels={'index': 'Columnas', 'value': 'Cantidad de Nulos'},
                 title='Cantidad de Valores Nulos por Columna',
                 text=null_counts.values)
    
    fig.update_traces(marker_color='royalblue')
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(title_x=0.5, title_font_size=20, 
                      xaxis_title_font_size=16, yaxis_title_font_size=16)
    fig.update_layout(xaxis_tickfont_size=14, yaxis_tickfont_size=14)
    fig.update_layout(bargap=0.2, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_yaxes(range=[0, max(null_counts.values) + 1])
    fig.update_xaxes(tickangle=45)
    
    return fig

def plot_distributions(df, selected_col=None):
    """
    Crea gráficos de distribución usando Plotly para variables numéricas.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        selected_col (str): Columna específica a graficar. Si es None, grafica todas.
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if selected_col:
        if selected_col not in numerical_cols:
            return None
        cols_to_plot = [selected_col]
    else:
        cols_to_plot = numerical_cols[:6]  # Limitar a 6 para mejor visualización
    
    fig = go.Figure()
    
    for col in cols_to_plot:
        fig.add_trace(go.Histogram(
            x=df[col].dropna(),
            name=col,
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        title='Distribuciones de Variables Numéricas',
        xaxis_title='Valor',
        yaxis_title='Frecuencia',
        barmode='overlay'
    )
    
    return fig

def plot_boxplots(df, selected_col=None):
    """
    Crea boxplots para variables numéricas usando Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        selected_col (str): Columna específica a graficar.
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if selected_col:
        if selected_col not in numerical_cols:
            return None
        cols_to_plot = [selected_col]
    else:
        cols_to_plot = numerical_cols[:6]
    
    fig = go.Figure()
    
    for i, col in enumerate(cols_to_plot):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title='Boxplots para Detección de Outliers',
        yaxis_title='Valor'
    )
    
    return fig

def plot_categorical_analysis(df, categorical_var, numerical_var=None):
    """
    Crea visualizaciones para variables categóricas usando Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        categorical_var (str): Nombre de la variable categórica a analizar.
        numerical_var (str, optional): Nombre de la variable numérica para comparación.
    """
    if categorical_var not in df.columns:
        return None
    
    # Gráfico de conteo
    counts = df[categorical_var].value_counts()
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        title=f'Distribución de {categorical_var}',
        labels={'x': categorical_var, 'y': 'Conteo'}
    )
    
    # Añadir porcentajes
    total = len(df)
    percentages = [f'{count/total*100:.1f}%' for count in counts.values]
    fig.update_traces(text=percentages, textposition='outside')
    
    return fig

def plot_pairplot(df, target, features=None):
    """
    Crea gráficos de dispersión para variables importantes usando Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        target (str): Nombre de la variable objetivo.
        features (list): Lista de características a incluir.
    """
    if features is None:
        if target not in df.columns:
            return None
        corr_with_target = df.corr()[target].abs().sort_values(ascending=False)
        features = corr_with_target.index[1:6].tolist()
    
    if len(features) < 2:
        return None
    
    fig = px.scatter_matrix(
        df,
        dimensions=features + [target],
        title=f'Matriz de Dispersión - Variables vs {target}'
    )
    
    return fig

def plot_scatter_with_size_color(df, x_var, y_var, color_var, size_var, title):
    """
    Crea un gráfico de dispersión con variables para los ejes X e Y,
    color y tamaño de los puntos.
    """
    fig = px.scatter(
        df,
        x=x_var,
        y=y_var,
        color=color_var,
        size=size_var,
        title=title,
        size_max=15,
        labels={x_var: x_var, y_var: y_var, color_var: color_var, size_var: size_var}
    )
    fig.update_layout(title_x=0.5, title_font_size=20)
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    return fig

def plot_correlation_matrix(df, figsize=(1000, 800)):
    """
    Crea una matriz de correlación para todas las variables numéricas usando Plotly.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        figsize (tuple): Tamaño de la figura en pixeles (ancho, alto).
    """
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numerical_cols].corr().round(2)
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale=px.colors.diverging.RdBu_r,
        zmin=-1, zmax=1,
        aspect="auto",
        labels=dict(color="Correlación"),
        x=corr_matrix.columns,
        y=corr_matrix.index,
        title="Matriz de Correlación"
    )
    fig.update_layout(
        width=figsize[0], height=figsize[1],
        title_x=0.5, title_font_size=24,
        xaxis_title="", yaxis_title="",
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig.update_xaxes(tickangle=45, side="top")
    fig.update_yaxes(autorange="reversed")
    return fig

def normalize_numerical_variables(df):
    """
    Normaliza las variables numéricas usando MinMaxScaler.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos originales
        
    Returns:
        tuple: (DataFrame normalizado, DataFrame original, scaler, columnas numéricas)
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col != 'ID']
    
    df_normalized = df.copy()
    
    scaler = MinMaxScaler()
    df_normalized[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_normalized, df, scaler, numerical_cols

def encode_categorical_variables(df):
    """
    Codifica las variables categóricas Education y MaritalStatus.
    
    Args:
        df (pandas.DataFrame): DataFrame con las variables a codificar
        
    Returns:
        pandas.DataFrame: DataFrame con las variables codificadas
    """
    df_encoded = df.copy()
    
    education_mapping = {
        '8thGrade': 1,
        '9-11thGrade': 2,
        'HighSchool': 3,
        'SomeCollege': 4,
        'CollegeGrad': 5
    }
    
    marital_mapping = {
        'NeverMarried': 1,
        'Married': 2,
        'Widowed': 3,
        'Divorced': 4,
        'Separated': 5,
        'LivePartner': 6
    }
    
    if 'Education' in df.columns:
        df_encoded['Education_Encoded'] = df['Education'].map(education_mapping)
    
    if 'MaritalStatus' in df.columns:
        df_encoded['MaritalStatus_Encoded'] = df['MaritalStatus'].map(marital_mapping)
    
    return df_encoded, education_mapping, marital_mapping

def plot_normalization_comparison(df_original, df_normalized, selected_col):
    """
    Crea visualización comparativa entre valores originales y normalizados.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_original[selected_col].dropna(),
        name='Original',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=df_normalized[selected_col].dropna(),
        name='Normalizado',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f'Comparación: {selected_col} - Original vs Normalizado',
        xaxis_title='Valor',
        yaxis_title='Frecuencia',
        barmode='overlay'
    )
    
    return fig

def create_derived_variables(df):
    """
    Crea variables derivadas combinando Age y Poverty.
    
    Args:
        df (pandas.DataFrame): DataFrame base
        
    Returns:
        pandas.DataFrame: DataFrame con variables derivadas
    """
    df_derived = df.copy()
    
    if 'Age' in df.columns and 'Poverty' in df.columns:
        age_bins = [0, 18, 35, 50, 65, 100]
        age_labels = ['Menor', 'Adulto_Joven', 'Adulto', 'Adulto_Mayor', 'Senior']
        df_derived['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
        
        def classify_socioeconomic(poverty_ratio):
            if pd.isna(poverty_ratio):
                return 'Desconocido'
            elif poverty_ratio <= 1:
                return 'Bajo'
            elif poverty_ratio <= 3:
                return 'Medio'
            else:
                return 'Alto'
        
        df_derived['Socioeconomic_Status'] = df['Poverty'].apply(classify_socioeconomic)
        df_derived['Income_Age'] = df_derived['Age_Group'].astype(str) + '_' + df_derived['Socioeconomic_Status']
        
        return df_derived, df_derived['Income_Age'].value_counts()
    else:
        return df, None

def plot_income_age_distribution(df, income_age_counts):
    """
    Crea un gráfico de barras para la distribución de Income_Age usando Plotly.
    """
    categories = income_age_counts.index.tolist()
    counts = income_age_counts.values.tolist()
    percentages = (income_age_counts / income_age_counts.sum() * 100).round(1).tolist()
    
    color_map = {
        'Alto': '#2E8B57',
        'Medio': '#4682B4',
        'Bajo': '#CD853F',
        'Desconocido': '#708090'
    }
    
    colors = []
    for category in categories:
        if 'Alto' in category:
            colors.append(color_map['Alto'])
        elif 'Medio' in category:
            colors.append(color_map['Medio'])
        elif 'Bajo' in category:
            colors.append(color_map['Bajo'])
        else:
            colors.append(color_map['Desconocido'])
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=counts,
            text=[f'{count}<br>({pct}%)' for count, pct in zip(counts, percentages)],
            textposition='outside',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>' +
                         'Conteo: %{y}<br>' +
                         'Porcentaje: %{text}<br>' +
                         '<extra></extra>',
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Distribución de Income_Age (Combinación Edad-Estatus Socioeconómico)',
            'x': 0.5,
            'font': {'size': 18, 'family': 'Arial', 'color': 'darkblue'}
        },
        xaxis_title='Categorías Income_Age',
        yaxis_title='Número de Participantes',
        font=dict(family='Arial, sans-serif', size=12, color='black'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=1000,
        height=600,
        margin=dict(l=50, r=50, t=80, b=120)
    )
    
    fig.update_xaxes(
        tickangle=45,
        tickfont=dict(size=10),
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def treat_outliers(df, method='remove'):
    """
    Trata los outliers en un DataFrame según el método especificado.
    
    Args:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        method (str): Método de tratamiento de outliers.
    
    Returns:
        pandas.DataFrame: DataFrame tratado según el método especificado.
    """
    if 'Outlier_LOF' not in df.columns:
        return df.copy()
        
    if method == 'remove':
        return df[df['Outlier_LOF'] != -1]
    elif method == 'replace':
        df_treated = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['Outlier_LOF', 'Outlier_IQR']:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_treated[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].mean(), df[col])
        return df_treated
    elif method == 'ignore':
        return df.copy()
    else:
        raise ValueError("Método no válido. Use 'remove', 'replace' o 'ignore'.")

def detect_outliers_lof(df, n_neighbors=20, contamination=0.05):
    """
    Detecta outliers en un DataFrame usando el algoritmo Local Outlier Factor (LOF).
    """
    # Seleccionar solo columnas numéricas
    numerical_data = df.select_dtypes(include=[np.number])
    
    # Verificar si hay datos disponibles
    if numerical_data.empty:
        return pd.Series(1, index=df.index)  # Retornar todos como normal si no hay datos numéricos
    
    # Eliminar filas con valores NaN
    clean_data = numerical_data.dropna()
    
    # Verificar si quedan datos después de eliminar NaN
    if len(clean_data) == 0:
        return pd.Series(1, index=df.index)  # Retornar todos como normal si no hay datos limpios
    
    # Verificar si quedan suficientes datos para LOF
    if len(clean_data) < n_neighbors:
        # Si no hay suficientes datos, ajustar n_neighbors o retornar valores por defecto
        n_neighbors = max(1, min(len(clean_data) - 1, 5))
        if n_neighbors < 1:
            return pd.Series(1, index=df.index)  # Retornar todos como normal
    
    # Verificar nuevamente después del ajuste
    if len(clean_data) < 2:  # LOF necesita al menos 2 muestras
        return pd.Series(1, index=df.index)  # Retornar todos como normal
    
    # Aplicar LOF solo a los datos limpios
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    clean_outlier_labels = lof.fit_predict(clean_data)
    
    # Crear serie de resultados para todo el DataFrame
    outlier_labels = pd.Series(1, index=df.index)  # Inicializar como normal (1)
    outlier_labels.loc[clean_data.index] = clean_outlier_labels
    
    return outlier_labels

def detect_outliers_iqr(df):
    """
    Detecta outliers en un DataFrame usando el método del rango intercuartílico (IQR).
    """
    outlier_flags = pd.Series(0, index=df.index)
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_flags |= ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
    
    return outlier_flags

def prepare_bmi_prediction_data(df):
    """
    Prepara los datos para la predicción de BMI.
    """
    feature_columns = [
        'Age', 'Weight', 'Height', 'Poverty', 'HomeRooms',
        'Pulse', 'BPSysAve', 'BPDiaAve', 'PhysActiveDays', 
        'SleepHrsNight', 'Education_Encoded', 'MaritalStatus_Encoded'
    ]
    
    available_features = [col for col in feature_columns if col in df.columns]
    df_model = df.dropna(subset=['BMI'] + available_features)
    
    X = df_model[available_features]
    y = df_model['BMI']
    
    return X, y, available_features

def train_bmi_model(X, y, feature_names):
    """
    Entrena modelos para predicción de BMI.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Linear_Regression': LinearRegression(),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        if name == 'Linear_Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_test': y_test,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'scaler': scaler if name == 'Linear_Regression' else None
        }
    
    return results, X_test, y_test

def prepare_diabetes_classification_data(df):
    """
    Prepara los datos para la clasificación de diabetes.
    """
    if 'Diabetes' not in df.columns:
        return None, None, None, None
    
    diabetes_counts = df['Diabetes'].value_counts()
    
    df_diabetes = df.copy()
    df_diabetes['Diabetes_Binary'] = (df_diabetes['Diabetes'] == 'Yes').astype(int)
    
    feature_columns = [
        'Age', 'BMI', 'Weight', 'Height', 'Poverty', 'BPSysAve', 'BPDiaAve',
        'Pulse', 'PhysActiveDays', 'SleepHrsNight', 'Education_Encoded', 
        'MaritalStatus_Encoded'
    ]
    
    available_features = [col for col in feature_columns if col in df_diabetes.columns]
    df_model = df_diabetes.dropna(subset=['Diabetes_Binary'] + available_features)
    
    X = df_model[available_features]
    y = df_model['Diabetes_Binary']
    
    class_counts = y.value_counts()
    
    return X, y, available_features, class_counts

def apply_undersampling_if_needed(X, y, threshold=0.3):
    """
    Aplica undersampling si el desbalance de clases es significativo.
    """
    class_counts = y.value_counts()
    minority_class = class_counts.min()
    majority_class = class_counts.max()
    
    imbalance_ratio = minority_class / majority_class
    
    if imbalance_ratio < threshold:
        df_combined = pd.concat([X, y], axis=1)
        
        df_majority = df_combined[df_combined.iloc[:, -1] == 0]
        df_minority = df_combined[df_combined.iloc[:, -1] == 1]
        
        df_majority_downsampled = resample(df_majority, 
                                         replace=False,
                                         n_samples=len(df_minority),
                                         random_state=42)
        
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
        
        X_balanced = df_balanced.iloc[:, :-1]
        y_balanced = df_balanced.iloc[:, -1]
        
        return X_balanced, y_balanced, True
    else:
        return X, y, False

def plot_class_distribution_comparison(original_counts, balanced_counts=None):
    """
    Crea gráficos comparativos de distribución de clases usando Plotly.
    """
    if balanced_counts is not None:
        labels = ['No Diabetes', 'Diabetes']
        
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=[original_counts[0], original_counts[1]],
            name="Original",
            domain=dict(x=[0, 0.45]),
            title="Distribución Original"
        ))
        
        fig.add_trace(go.Pie(
            labels=labels,
            values=[balanced_counts[0], balanced_counts[1]],
            name="Balanceado",
            domain=dict(x=[0.55, 1]),
            title="Distribución Balanceada"
        ))
        
        fig.update_layout(title_text="Comparación de Distribución de Clases")
        
    else:
        labels = ['No Diabetes', 'Diabetes']
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=[original_counts[0], original_counts[1]]
        )])
        fig.update_layout(title_text="Distribución de Clases - Diabetes")
    
    return fig

def train_neural_network_diabetes(X, y, feature_names):
    """
    Entrena una red neuronal para clasificación de diabetes.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    y_pred = mlp.predict(X_test_scaled)
    y_pred_proba = mlp.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    
    results = {
        'model': mlp,
        'scaler': scaler,
        'y_pred': y_pred,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_names': feature_names
    }
    
    return results

def plot_confusion_matrix(conf_matrix, class_names=['No Diabetes', 'Diabetes']):
    """
    Crea un gráfico de matriz de confusión usando Plotly.
    """
    fig = ff.create_annotated_heatmap(
        z=conf_matrix,
        x=class_names,
        y=class_names,
        annotation_text=conf_matrix,
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title='Matriz de Confusión - Clasificación de Diabetes',
        xaxis_title='Predicción',
        yaxis_title='Valor Real',
        font=dict(size=12),
        width=600,
        height=500
    )
    
    fig['layout']['yaxis']['autorange'] = 'reversed'
    
    return fig

def clean_nhanes_data(df):
    """
    Pipeline completo de limpieza y transformación de datos NHANES.
    
    Args:
        df (pandas.DataFrame): DataFrame original de NHANES
        
    Returns:
        pandas.DataFrame: DataFrame limpio y transformado
    """
    df_clean = df.copy()
    
    # 1. Eliminar variables redundantes
    columns_to_drop = ['AgeDecade', 'AgeMonths', 'Race1', 'HHIncome', 'BMI_WHO',
                      'BPSys1', 'BPSys2', 'BPSys3', 'BPDia1', 'BPDia2', 'BPDia3',
                      'nPregnancies', 'Alcohol12PlusYr', 'AgeFirstMarij', 'HHIncomeMid']
    
    columns_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    df_clean.drop(columns=columns_to_drop, inplace=True)
    
    # 2. Reemplazar valores especiales con NaN
    df_clean.replace({9999: pd.NA, 7777: pd.NA, 'Refused': pd.NA}, inplace=True)
    
    # 3. Eliminar columnas con más del 80% de nulos
    df_clean.dropna(thresh=len(df_clean) * 0.8, axis=1, inplace=True)
    
    # 4. Imputar valores nulos
    for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
    for col in df_clean.select_dtypes(include=['object']).columns:
        if len(df_clean[col].mode()) > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # 5. Detectar outliers
    df_clean['Outlier_LOF'] = detect_outliers_lof(df_clean)
    df_clean['Outlier_IQR'] = detect_outliers_iqr(df_clean)
    
    # 6. Tratar outliers (eliminar)
    df_clean = treat_outliers(df_clean, method='remove')
    df_clean = df_clean.drop(['Outlier_LOF', 'Outlier_IQR'], axis=1, errors='ignore')
    
    # 7. Normalizar variables numéricas
    df_normalized, _, _, _ = normalize_numerical_variables(df_clean)
    
    # 8. Codificar variables categóricas
    df_encoded, _, _ = encode_categorical_variables(df_normalized)
    
    # 9. Crear variables derivadas
    df_final, _ = create_derived_variables(df_encoded)
    
    return df_final