# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------
# 1. Load Data
# --------------------
st.title("📊 High-Level & Interpretable EDA Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --------------------
    # 2. Basic Info
    # --------------------
    st.subheader("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("Column names:", df.columns.tolist())

    st.write("Data Types:")
    st.write(df.dtypes)

    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # --------------------
    # 3. Summary Stats
    # --------------------
    st.subheader("Summary Statistics")
    st.write("Numeric Columns")
    st.write(df.describe())

    st.write("Categorical Columns")
    st.write(df.describe(include="object"))

    # --------------------
    # 4. Correlation Heatmap
    # --------------------
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="viridis", ax=ax)
        st.pyplot(fig)

    # --------------------
    # 5. Distribution of Variables
    # --------------------
    st.subheader("Distribution of Numeric Columns")
    column = st.selectbox("Select a numeric column", numeric_df.columns if not numeric_df.empty else [])
    if column:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax)
        st.pyplot(fig)

    # --------------------
    # 6. Boxplot for Outliers
    # --------------------
    st.subheader("Outlier Detection (Boxplot)")
    if column:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        st.pyplot(fig)

    # --------------------
    # 7. Categorical Analysis
    # --------------------
    st.subheader("Categorical Analysis")
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        cat_col = st.selectbox("Select a categorical column", cat_cols)
        fig, ax = plt.subplots()
        df[cat_col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Frequency of {cat_col}")
        st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to start the analysis.")
