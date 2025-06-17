import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Dataset Relationship Visualizer", layout="wide")

st.title("📊 Data Relationship Visualizer")

st.write(
    """
    Upload your dataset (CSV file), and explore the relationships between features with automated visualizations.
    """
)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Dataset Information")
    st.write(f"Shape: {df.shape}")
    st.write("Column Types:")
    st.write(df.dtypes)

    # ================================
    st.subheader("📍 Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    st.pyplot(fig)

    # ================================
    st.subheader("🔗 Correlation Heatmap (Numerical Columns)")
    numeric_cols = df.select_dtypes(include=np.number)
    if numeric_cols.shape[1] >= 2:
        corr = numeric_cols.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for correlation heatmap.")

    # ================================
    st.subheader("📌 Pairplot of Numerical Features")
    if numeric_cols.shape[1] >= 2:
        st.info("Generating pairplot (can take time for large datasets)...")
        fig = sns.pairplot(numeric_cols)
        st.pyplot(fig)
    else:
        st.info("Not enough numerical columns for pairplot.")

    # ================================
    st.subheader("🌀 Explore Relationships")

    columns = df.columns.tolist()
    x_axis = st.selectbox("Select X-axis", columns)
    y_axis = st.selectbox("Select Y-axis", columns)
    color_by = st.selectbox("Color by (Optional)", [None] + columns)

    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by)
    st.plotly_chart(fig, use_container_width=True)

    # ================================
    st.subheader("📂 Categorical Relationships")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        cat_col = st.selectbox("Select categorical column for analysis", categorical_cols)
        fig = px.box(df, x=cat_col, y=numeric_cols.columns[0], color=cat_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No categorical columns found.")

else:
    st.warning("Upload a CSV file to start exploring!")