import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import utils as ut

# Configure Streamlit page
st.set_page_config(
    page_title="NHANES 2009-2012 Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c5aa0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e8f4fd;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .data-dict-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2c5aa0;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the NHANES dataset"""
    import os
    
    # Try multiple possible paths
    possible_paths = [
        "../Data/NHANES2009-2012.csv",
        "Data/NHANES2009-2012.csv", 
        "/home/andersonjalvarado/Sync/AIM/Gestion_Datos/Trabajos_Gestion_Datos/Taller_1/Data/NHANES2009-2012.csv"
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Datos cargados exitosamente desde: {path}")
                return df
        except Exception as e:
            continue
    
    st.error("❌ No se pudo encontrar el archivo de datos. Ubicaciones verificadas:")
    for path in possible_paths:
        st.error(f"   - {path}")
    st.info("💡 Asegúrese de que 'NHANES2009-2012.csv' esté en alguna de estas ubicaciones.")
    return None

def create_data_dictionary():
    """Create and display data dictionary section"""
    st.markdown('<h2 class="section-header">📚 Diccionario de Datos</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 Acerca del Dataset NHANES 2009-2012</h4>
        <p>El <strong>National Health and Nutrition Examination Survey (NHANES)</strong> es una encuesta representativa a nivel nacional que evalúa el estado de salud y nutrición de adultos y niños en Estados Unidos. Los datos corresponden al período 2009-2012.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create data dictionary with categories
    data_dict = {
        "🏥 Variables Demográficas": [
            ("ID", "Identificador único del participante", "Numérico"),
            ("Age", "Edad en años", "18-80 años"),
            ("AgeDecade", "Década de edad", "Categórico"),
            ("Gender", "Género del participante", "Male/Female"),
            ("Race1", "Raza/Etnia", "Categórico"),
            ("Education", "Nivel educativo", "8thGrade, 9-11thGrade, HighSchool, SomeCollege, CollegeGrad"),
            ("MaritalStatus", "Estado civil", "Married, Widowed, Divorced, Separated, NeverMarried, LivePartner")
        ],
        "💰 Variables Socioeconómicas": [
            ("HHIncome", "Ingreso familiar anual", "Categórico por rangos"),
            ("HHIncomeMid", "Punto medio del ingreso familiar", "Numérico USD"),
            ("Poverty", "Ratio de pobreza", "0-5+ (1=línea de pobreza)"),
            ("HomeRooms", "Número de habitaciones en el hogar", "1-13")
        ],
        "📏 Variables Físicas": [
            ("Weight", "Peso corporal", "Kilogramos"),
            ("Height", "Altura", "Centímetros"),
            ("BMI", "Índice de Masa Corporal", "kg/m²"),
            ("BMI_WHO", "Categoría BMI según OMS", "Underweight, Normal, Overweight, Obese")
        ],
        "❤️ Variables de Salud Cardiovascular": [
            ("Pulse", "Frecuencia cardíaca", "Latidos por minuto"),
            ("BPSysAve", "Presión arterial sistólica promedio", "mmHg"),
            ("BPDiaAve", "Presión arterial diastólica promedio", "mmHg"),
            ("BPSys1/2/3", "Lecturas individuales de presión sistólica", "mmHg"),
            ("BPDia1/2/3", "Lecturas individuales de presión diastólica", "mmHg")
        ],
        "🏃 Variables de Estilo de Vida": [
            ("Smoke100", "¿Ha fumado 100+ cigarrillos?", "Yes/No"),
            ("SmokeAge", "Edad al comenzar a fumar", "Años"),
            ("Marijuana", "¿Ha usado marihuana?", "Yes/No"),
            ("AgeFirstMarij", "Edad del primer uso de marihuana", "Años"),
            ("PhysActive", "¿Físicamente activo?", "Yes/No"),
            ("PhysActiveDays", "Días de actividad física por semana", "0-7"),
            ("AlcoholDay", "Bebidas alcohólicas por día", "Número"),
            ("AlcoholYear", "Días de consumo de alcohol por año", "0-365"),
            ("SleepHrsNight", "Horas de sueño por noche", "2-12")
        ],
        "🏥 Variables de Salud": [
            ("Diabetes", "Estado de diabetes", "Yes/No"),
            ("DiabetesAge", "Edad al diagnóstico de diabetes", "Años"),
            ("HealthGen", "Salud general autorreportada", "Excellent, Vgood, Good, Fair, Poor"),
            ("DaysPhysHlthBad", "Días de mala salud física (último mes)", "0-30"),
            ("DaysMentHlthBad", "Días de mala salud mental (último mes)", "0-30"),
            ("LittleInterest", "Poco interés en actividades", "None, Several, Majority, AlmostAll"),
            ("Depressed", "Sentimientos de depresión", "None, Several, Majority, AlmostAll")
        ],
        "🤱 Variables Reproductivas": [
            ("nPregnancies", "Número de embarazos", "0-32"),
            ("nBabies", "Número de bebés nacidos vivos", "0-12"),
            ("SexAge", "Edad de la primera relación sexual", "8-69")
        ]
    }

    # Display data dictionary in organized cards
    for category, variables in data_dict.items():
        st.markdown(f"### {category}")
        
        for var_name, description, values in variables:
            st.markdown(f"""
            <div class="data-dict-card">
                <strong>{var_name}</strong>: {description}<br>
                <em>Valores: {values}</em>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 NHANES 2009-2012 Health Data Dashboard</h1>', unsafe_allow_html=True)
    
    # Header layout with date filter
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🔍 Análisis Exploratorio de Datos de Salud y Nutrición")
        st.markdown("*National Health and Nutrition Examination Survey (2009-2012)*")
    
    with col2:
        st.markdown("#### 📅 Filtro de Período")
        start_date = st.date_input(
            "Fecha inicio",
            value=date(2009, 1, 1),
            min_value=date(2009, 1, 1),
            max_value=date(2012, 12, 31)
        )
        end_date = st.date_input(
            "Fecha fin",
            value=date(2012, 12, 31),
            min_value=date(2009, 1, 1),
            max_value=date(2012, 12, 31)
        )
        
        # Note: Date filters are displayed for UI completeness
        # The NHANES dataset covers the entire 2009-2012 period
        if start_date and end_date:
            st.info(f"📊 Período seleccionado: {start_date} - {end_date}")

    # Load data
    df_original = load_data()
    if df_original is None:
        return

    # Sidebar navigation
    st.sidebar.title("🧭 Navegación")
    sections = [
        "📚 Diccionario de Datos",
        "📊 Resumen y Métricas",
        "🔍 Exploración de Datos", 
        "⚙️ Preparación y Transformación",
        "🤖 Modelado Predictivo"
    ]
    
    selected_section = st.sidebar.radio("Seleccione una sección:", sections)

    # Display selected section
    if selected_section == "📚 Diccionario de Datos":
        create_data_dictionary()
        
    elif selected_section == "📊 Resumen y Métricas":
        create_overview_metrics_section(df_original)
        
    elif selected_section == "🔍 Exploración de Datos":
        create_data_exploration_section(df_original)
        
    elif selected_section == "⚙️ Preparación y Transformación":
        create_preparation_section(df_original)
        
    elif selected_section == "🤖 Modelado Predictivo":
        create_modeling_section(df_original)

def create_overview_metrics_section(df):
    """Create Overview & Metrics section"""
    st.markdown('<h2 class="section-header">📊 Resumen y Métricas</h2>', unsafe_allow_html=True)
    
    # Dataset overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-label">Total Participantes</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Variables Totales</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        num_cols = len(df.select_dtypes(include=np.number).columns)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{num_cols}</div>
            <div class="metric-label">Variables Numéricas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        cat_cols = len(df.select_dtypes(include='object').columns)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{cat_cols}</div>
            <div class="metric-label">Variables Categóricas</div>
        </div>
        """, unsafe_allow_html=True)

    # Column distribution chart
    st.subheader("📈 Distribución de Tipos de Columnas")
    fig_dist = ut.plot_distributions_cant(df)
    st.plotly_chart(fig_dist, use_container_width=True)

    # DataFrame analysis
    st.subheader("🔍 Análisis del DataFrame")
    analysis_results = ut.analize_dataframe(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Información General:**")
        st.write(f"- Forma del dataset: {analysis_results['shape']}")
        st.write(f"- Total de valores nulos: {analysis_results['total_nulos']:,}")
        
        st.write("**Información de Duplicados:**")
        st.dataframe(analysis_results['duplicados'], use_container_width=True)
    
    with col2:
        st.write("**Valores Nulos por Columna:**")
        st.dataframe(analysis_results['nulos_por_columna'].head(10), use_container_width=True)

    # Basic statistics
    st.subheader("📈 Estadísticas Básicas")
    basic_stats = ut.basic_statistics(df)
    st.dataframe(basic_stats, use_container_width=True)

def create_data_exploration_section(df):
    """Create Data Exploration section"""
    st.markdown('<h2 class="section-header">🔍 Exploración de Datos</h2>', unsafe_allow_html=True)
    
    # Null values visualization
    st.subheader("🕳️ Valores Nulos")
    fig_nulls = ut.plot_nulls(df)
    st.plotly_chart(fig_nulls, use_container_width=True)
    
    # Distribution analysis
    st.subheader("📊 Análisis de Distribuciones")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numerical_cols:
        selected_num_col = st.selectbox("Seleccione variable numérica:", numerical_cols)
        if selected_num_col:
            fig_dist = ut.plot_distributions(df, selected_num_col)
            if fig_dist:
                st.plotly_chart(fig_dist, use_container_width=True)
    
    # Boxplot analysis
    st.subheader("📦 Análisis de Outliers (Boxplots)")
    if numerical_cols:
        selected_box_col = st.selectbox("Seleccione variable para boxplot:", numerical_cols, key="boxplot")
        if selected_box_col:
            fig_box = ut.plot_boxplots(df, selected_box_col)
            if fig_box:
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Categorical analysis
    st.subheader("📋 Análisis de Variables Categóricas")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    
    if categorical_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_cat_col = st.selectbox("Variable categórica:", categorical_cols)
        with col2:
            selected_num_for_cat = st.selectbox("Variable numérica (opcional):", [None] + numerical_cols)
        
        if selected_cat_col:
            fig_cat = ut.plot_categorical_analysis(df, selected_cat_col, selected_num_for_cat)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
    
    # Pairplot analysis
    st.subheader("🔗 Relaciones entre Variables (Pairplot)")
    if len(numerical_cols) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            target_var = st.selectbox("Variable objetivo:", numerical_cols)
        with col2:
            feature_vars = st.multiselect("Variables a comparar:", numerical_cols, 
                                        default=numerical_cols[:min(3, len(numerical_cols))])
        
        if target_var and feature_vars and target_var in df.columns:
            fig_pair = ut.plot_pairplot(df, target_var, feature_vars)
            if fig_pair:
                st.plotly_chart(fig_pair, use_container_width=True)
    
    # Scatter plot with multiple dimensions
    st.subheader("🎯 Gráfico de Dispersión Multidimensional")
    if len(numerical_cols) >= 4:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_var = st.selectbox("Eje X:", numerical_cols, key="scatter_x")
        with col2:
            y_var = st.selectbox("Eje Y:", numerical_cols, key="scatter_y")
        with col3:
            color_var = st.selectbox("Color:", numerical_cols, key="scatter_color")
        with col4:
            size_var = st.selectbox("Tamaño:", numerical_cols, key="scatter_size")
        
        if all([x_var, y_var, color_var, size_var]):
            title = f"Scatter Plot: {y_var} vs {x_var}"
            fig_scatter = ut.plot_scatter_with_size_color(df, x_var, y_var, color_var, size_var, title)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Correlation matrix
    st.subheader("🔄 Matriz de Correlaciones")
    fig_corr = ut.plot_correlation_matrix(df)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Outlier detection tables
    st.subheader("🎯 Detección de Outliers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Outliers - Local Outlier Factor (LOF)**")
        outliers_lof = ut.detect_outliers_lof(df)
        lof_outliers = sum(outliers_lof == -1)
        st.metric("Outliers detectados (LOF)", f"{lof_outliers:,}")
        st.write(f"Porcentaje de outliers: {lof_outliers/len(df)*100:.2f}%")
    
    with col2:
        st.write("**Outliers - Rango Intercuartílico (IQR)**")
        outliers_iqr = ut.detect_outliers_iqr(df)
        iqr_outliers = sum(outliers_iqr == 1)
        st.metric("Outliers detectados (IQR)", f"{iqr_outliers:,}")
        st.write(f"Porcentaje de outliers: {iqr_outliers/len(df)*100:.2f}%")

def create_preparation_section(df):
    """Create Preparation & Transformation section"""
    st.markdown('<h2 class="section-header">⚙️ Preparación y Transformación</h2>', unsafe_allow_html=True)
    
    st.info("ℹ️ **Nota importante**: Las visualizaciones anteriores utilizan datos originales. En esta sección aplicamos las transformaciones que se usarán en el modelado.")
    
    # Normalization comparison
    st.subheader("📏 Normalización de Variables Numéricas")
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'ID' in numerical_cols:
        numerical_cols.remove('ID')
    
    if numerical_cols:
        selected_norm_col = st.selectbox("Seleccione variable para comparar normalización:", numerical_cols)
        
        if selected_norm_col:
            df_normalized, _, _, _ = ut.normalize_numerical_variables(df)
            fig_norm = ut.plot_normalization_comparison(df, df_normalized, selected_norm_col)
            st.plotly_chart(fig_norm, use_container_width=True)
            
            # Show normalization stats
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Estadísticas Originales:**")
                orig_stats = df[selected_norm_col].describe()
                st.write(orig_stats)
            
            with col2:
                st.write("**Estadísticas Normalizadas:**")
                norm_stats = df_normalized[selected_norm_col].describe()
                st.write(norm_stats)
    
    # Categorical encoding
    st.subheader("🏷️ Codificación de Variables Categóricas")
    
    df_encoded, education_map, marital_map = ut.encode_categorical_variables(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Education' in df.columns:
            st.write("**Codificación de Education:**")
            for key, value in education_map.items():
                count = df['Education'].value_counts().get(key, 0)
                st.write(f"• {key}: {value} (n={count:,})")
    
    with col2:
        if 'MaritalStatus' in df.columns:
            st.write("**Codificación de MaritalStatus:**")
            for key, value in marital_map.items():
                count = df['MaritalStatus'].value_counts().get(key, 0)
                st.write(f"• {key}: {value} (n={count:,})")
    
    # Derived variables
    st.subheader("🔧 Creación de Variables Derivadas")
    
    df_derived, income_age_dist = ut.create_derived_variables(df_encoded)
    
    if income_age_dist is not None:
        st.write("**Variable Income_Age creada exitosamente**")
        st.write("Combina grupos de edad con estatus socioeconómico basado en el ratio de pobreza.")
        
        fig_income_age = ut.plot_income_age_distribution(df_derived, income_age_dist)
        st.plotly_chart(fig_income_age, use_container_width=True)
        
        # Show distribution table
        st.write("**Distribución de Income_Age:**")
        income_age_df = income_age_dist.to_frame('Conteo')
        income_age_df['Porcentaje'] = (income_age_dist / income_age_dist.sum() * 100).round(2)
        st.dataframe(income_age_df, use_container_width=True)
    
    # Outlier treatment
    st.subheader("🎯 Tratamiento de Outliers")
    
    treatment_method = st.selectbox(
        "Seleccione método de tratamiento:",
        ['remove', 'replace', 'ignore'],
        format_func=lambda x: {
            'remove': 'Eliminar outliers',
            'replace': 'Reemplazar con media',
            'ignore': 'No tratar'
        }[x]
    )
    
    # Add outlier detection columns for treatment
    df_with_outliers = df.copy()
    df_with_outliers['Outlier_LOF'] = ut.detect_outliers_lof(df)
    df_with_outliers['Outlier_IQR'] = ut.detect_outliers_iqr(df)
    
    df_treated = ut.treat_outliers(df_with_outliers, method=treatment_method)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Filas originales", f"{len(df):,}")
    with col2:
        st.metric("Filas después del tratamiento", f"{len(df_treated):,}")
    
    # Final clean dataset
    st.subheader("🧹 Dataset Final Limpio")
    
    with st.expander("Ver función clean_nhanes_data"):
        st.code('''
def clean_nhanes_data(df):
    """
    Pipeline completo de limpieza y transformación de datos NHANES.
    
    Pasos:
    1. Eliminar variables redundantes
    2. Reemplazar valores especiales con NaN
    3. Eliminar columnas con >80% de nulos
    4. Imputar valores nulos
    5. Detectar outliers
    6. Tratar outliers
    7. Normalizar variables numéricas
    8. Codificar variables categóricas
    9. Crear variables derivadas
    
    Returns:
        pandas.DataFrame: Dataset limpio y transformado
    """
        ''')
    
    if st.button("🔄 Aplicar Pipeline Completo de Limpieza"):
        with st.spinner("Aplicando transformaciones..."):
            df_final_clean = ut.clean_nhanes_data(df)
            
            st.success("✅ Pipeline de limpieza aplicado exitosamente")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Filas originales", f"{len(df):,}")
            with col2:
                st.metric("Filas finales", f"{len(df_final_clean):,}")
            with col3:
                reduction = (1 - len(df_final_clean)/len(df)) * 100
                st.metric("Reducción", f"{reduction:.1f}%")
            
            st.write("**Vista previa del dataset limpio:**")
            st.dataframe(df_final_clean.head(), use_container_width=True)
            
            # Store cleaned data in session state for modeling section
            st.session_state['df_clean'] = df_final_clean

def create_modeling_section(df):
    """Create Predictive Modeling section"""
    st.markdown('<h2 class="section-header">🤖 Modelado Predictivo</h2>', unsafe_allow_html=True)
    
    # Check if we have cleaned data, if not, create it
    if 'df_clean' not in st.session_state:
        st.info("⚠️ Aplicando pipeline de limpieza automáticamente...")
        with st.spinner("Limpiando datos..."):
            df_clean = ut.clean_nhanes_data(df)
            st.session_state['df_clean'] = df_clean
    else:
        df_clean = st.session_state['df_clean']
    
    st.success(f"✅ Usando dataset limpio con {len(df_clean):,} registros")
    
    # BMI Prediction
    st.subheader("📊 Predicción de BMI (Regresión)")
    
    with st.expander("ℹ️ Información sobre el modelo de BMI"):
        st.write("""
        **Objetivo**: Predecir el Índice de Masa Corporal (BMI) basado en características demográficas, socioeconómicas y de estilo de vida.
        
        **Características utilizadas**:
        - Age, Weight, Height, Poverty, HomeRooms
        - Pulse, BPSysAve, BPDiaAve, PhysActiveDays
        - SleepHrsNight, Education_Encoded, MaritalStatus_Encoded
        
        **Modelos**: Linear Regression y Random Forest
        """)
    
    if st.button("🏃 Entrenar Modelos de BMI"):
        with st.spinner("Entrenando modelos de BMI..."):
            try:
                X_bmi, y_bmi, bmi_features = ut.prepare_bmi_prediction_data(df_clean)
                
                if X_bmi is not None and len(X_bmi) > 0:
                    bmi_results, _, _ = ut.train_bmi_model(X_bmi, y_bmi, bmi_features)
                    
                    st.success("✅ Modelos de BMI entrenados exitosamente")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    for i, (model_name, results) in enumerate(bmi_results.items()):
                        with col1 if i == 0 else col2:
                            st.write(f"**{model_name}**")
                            st.metric("RMSE", f"{results['rmse']:.3f}")
                            st.metric("MAE", f"{results['mae']:.3f}")
                            st.metric("R²", f"{results['r2']:.3f}")
                    
                    # Feature importance for Random Forest
                    if 'Random_Forest' in bmi_results:
                        rf_model = bmi_results['Random_Forest']['model']
                        importance_df = pd.DataFrame({
                            'Feature': bmi_features,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.write("**Importancia de Características (Random Forest):**")
                        st.dataframe(importance_df, use_container_width=True)
                
                else:
                    st.error("❌ No se pudieron preparar los datos para predicción de BMI")
                    
            except Exception as e:
                st.error(f"❌ Error al entrenar modelos de BMI: {str(e)}")
    
    # Diabetes Classification
    st.subheader("🩺 Clasificación de Diabetes")
    
    with st.expander("ℹ️ Información sobre el modelo de Diabetes"):
        st.write("""
        **Objetivo**: Clasificar el estado de diabetes (Sí/No) usando una red neuronal.
        
        **Características utilizadas**:
        - Age, BMI, Weight, Height, Poverty
        - BPSysAve, BPDiaAve, Pulse, PhysActiveDays
        - SleepHrsNight, Education_Encoded, MaritalStatus_Encoded
        
        **Modelo**: Multi-Layer Perceptron (Red Neuronal)
        **Arquitectura**: 2 capas ocultas (100, 50 neuronas)
        """)
    
    if st.button("🧠 Entrenar Red Neuronal para Diabetes"):
        with st.spinner("Entrenando modelo de diabetes..."):
            try:
                X_diabetes, y_diabetes, diabetes_features, original_class_counts = ut.prepare_diabetes_classification_data(df_clean)
                
                if X_diabetes is not None and len(X_diabetes) > 0:
                    # Show class distribution
                    st.write("**Distribución de Clases Original:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No Diabetes", f"{original_class_counts[0]:,}")
                    with col2:
                        st.metric("Diabetes", f"{original_class_counts[1]:,}")
                    
                    # Apply undersampling if needed
                    X_diabetes_balanced, y_diabetes_balanced, undersampling_applied = ut.apply_undersampling_if_needed(X_diabetes, y_diabetes)
                    
                    if undersampling_applied:
                        st.info("ℹ️ Se aplicó undersampling para balancear las clases")
                        balanced_class_counts = y_diabetes_balanced.value_counts()
                        
                        # Show class distribution comparison
                        fig_class_dist = ut.plot_class_distribution_comparison(original_class_counts, balanced_class_counts)
                        st.plotly_chart(fig_class_dist, use_container_width=True)
                        
                        # Train with balanced data
                        diabetes_results = ut.train_neural_network_diabetes(X_diabetes_balanced, y_diabetes_balanced, diabetes_features)
                    else:
                        # Train with original data
                        diabetes_results = ut.train_neural_network_diabetes(X_diabetes, y_diabetes, diabetes_features)
                    
                    st.success("✅ Modelo de diabetes entrenado exitosamente")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Precisión", f"{diabetes_results['accuracy']:.3f}")
                        st.metric("Iteraciones", diabetes_results['model'].n_iter_)
                    
                    with col2:
                        st.write("**Matriz de Confusión:**")
                        conf_matrix = diabetes_results['confusion_matrix']
                        st.write(f"True Negatives: {conf_matrix[0,0]}")
                        st.write(f"False Positives: {conf_matrix[0,1]}")
                        st.write(f"False Negatives: {conf_matrix[1,0]}")
                        st.write(f"True Positives: {conf_matrix[1,1]}")
                    
                    # Show confusion matrix plot
                    fig_conf_matrix = ut.plot_confusion_matrix(diabetes_results['confusion_matrix'])
                    st.plotly_chart(fig_conf_matrix, use_container_width=True)
                    
                    # Classification report
                    st.write("**Reporte de Clasificación:**")
                    st.text(diabetes_results['classification_report'])
                
                else:
                    st.error("❌ No se pudieron preparar los datos para clasificación de diabetes")
                    
            except Exception as e:
                st.error(f"❌ Error al entrenar modelo de diabetes: {str(e)}")

if __name__ == "__main__":
    main()