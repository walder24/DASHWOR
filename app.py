import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris

# --------------------------------------------------
# Configuración de la página
# --------------------------------------------------
st.set_page_config(
    page_title="Dashboard Iris",
    page_icon="🌸",
    layout="wide"
)

# --------------------------------------------------
# Cargar datos
# --------------------------------------------------
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(
        dict(enumerate(iris.target_names))
    )
    return df

df = load_data()

# --------------------------------------------------
# Título
# --------------------------------------------------
st.title("🌸 Dashboard Dataset Iris")
st.markdown(
    "Análisis exploratorio del conjunto de datos Iris utilizando Streamlit y Plotly."
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Filtros")

species = st.sidebar.multiselect(
    "Seleccionar especies",
    options=df["species"].unique(),
    default=df["species"].unique()
)

filtered_df = df[df["species"].isin(species)]

# --------------------------------------------------
# Métricas
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Observaciones", len(filtered_df))
col2.metric("Variables", 4)
col3.metric("Especies", filtered_df["species"].nunique())
col4.metric(
    "Prom. Sepal Length",
    round(filtered_df["sepal length (cm)"].mean(), 2)
)

st.divider()

# --------------------------------------------------
# Gráficos principales
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    fig_scatter = px.scatter(
        filtered_df,
        x="sepal length (cm)",
        y="petal length (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Sepal Length vs Petal Length",
        height=500
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    fig_box = px.box(
        filtered_df,
        x="species",
        y="sepal width (cm)",
        color="species",
        color_discrete_sequence=px.colors.sequential.Viridis,
        title="Distribución Sepal Width",
        height=500
    )

    st.plotly_chart(fig_box, use_container_width=True)

# --------------------------------------------------
# Histograma
# --------------------------------------------------
fig_hist = px.histogram(
    filtered_df,
    x="petal width (cm)",
    color="species",
    nbins=20,
    color_discrete_sequence=px.colors.sequential.Viridis,
    title="Distribución de Petal Width"
)

st.plotly_chart(fig_hist, use_container_width=True)

# --------------------------------------------------
# Matriz de correlación
# --------------------------------------------------
st.subheader("Matriz de Correlación")

corr = filtered_df.drop(
    columns=["target", "species"]
).corr()

fig_corr = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="Viridis",
    aspect="auto"
)

st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# Datos
# --------------------------------------------------
st.subheader("Datos del Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# --------------------------------------------------
# Resumen estadístico
# --------------------------------------------------
st.subheader("Resumen Estadístico")

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)
