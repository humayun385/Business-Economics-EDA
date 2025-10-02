import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("📊 High-Level EDA Dashboard")
st.write("Exploratory Data Analysis with Streamlit")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Dataset preview
    st.subheader("🔎 Dataset Preview")
    st.dataframe(df.head())

    # Basic Info
    st.subheader("📑 Dataset Information")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("Column Names:", list(df.columns))

    # Summary Statistics
    st.subheader("📈 Summary Statistics (Numeric Columns)")
    st.write(df.describe())

    # Drop ID-like columns before correlation
    id_columns = ['order_id', 'customer_id', 'product_id']
    df_corr = df.drop(columns=[col for col in id_columns if col in df.columns], errors="ignore")

    # Select numeric & categorical columns
    numeric_df = df_corr.select_dtypes(include=['int64', 'float64'])
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Correlation Heatmap
    if not numeric_df.empty:
        st.subheader("🔗 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="viridis", ax=ax, fmt=".2f")
        st.pyplot(fig)

    # Distribution plots
    st.subheader("📊 Distribution of Numeric Features")
    if len(numeric_df.columns) > 0:
        col_choice = st.selectbox("Select column for histogram:", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col_choice], kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)

    # Count plots for categorical features
    st.subheader("🧾 Value Counts for Categorical Features")
    if len(categorical_cols) > 0:
        cat_choice = st.selectbox("Select column for count plot:", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=cat_choice, data=df, palette="Set2", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Boxplots (Outlier detection)
    st.subheader("📦 Boxplot (Outlier Detection)")
    if len(numeric_df.columns) > 0:
        box_choice = st.selectbox("Select numeric column for boxplot:", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.boxplot(x=df[box_choice], color="orange", ax=ax)
        st.pyplot(fig)

    # Pairplot (Relationships between numeric variables)
    if len(numeric_df.columns) > 1:
        st.subheader("📌 Pairwise Relationships (Pairplot)")
        st.write("Showing relationships between numeric columns")
        fig = sns.pairplot(numeric_df, diag_kind="kde", corner=True, palette="husl")
        st.pyplot(fig)

    # Scatter plot (between two numeric variables)
    st.subheader("🌐 Scatter Plot")
    if len(numeric_df.columns) > 1:
        x_axis = st.selectbox("Select X-axis:", numeric_df.columns, index=0)
        y_axis = st.selectbox("Select Y-axis:", numeric_df.columns, index=1)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_axis], y=df[y_axis], alpha=0.7, s=60, color="green", ax=ax)
        st.pyplot(fig)

    # Bar chart for aggregated values
    st.subheader("📊 Aggregated Bar Chart")
    if len(categorical_cols) > 0 and len(numeric_df.columns) > 0:
        agg_cat = st.selectbox("Select category:", categorical_cols)
        agg_num = st.selectbox("Select numeric column:", numeric_df.columns)
        fig, ax = plt.subplots()
        df.groupby(agg_cat)[agg_num].mean().sort_values(ascending=False).plot(kind="bar", ax=ax, color="teal")
        plt.xticks(rotation=45)
        st.pyplot(fig)
