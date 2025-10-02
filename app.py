import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
st.title("📊 High-Level EDA Dashboard")
st.write("Exploratory Data Analysis with Streamlit")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    
    # Show dataset preview
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

    # Select only numeric columns for correlation
    numeric_df = df_corr.select_dtypes(include=['int64', 'float64'])

    # Correlation Heatmap
    if not numeric_df.empty:
        st.subheader("🔗 Correlation Heatmap (Numeric features only, no IDs)")
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="viridis", ax=ax, fmt=".2f")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns available for correlation after removing IDs.")

    # Distribution of numeric columns
    st.subheader("📊 Distribution of Numeric Features")
    if len(numeric_df.columns) > 0:
        col_choice = st.selectbox("Select column for distribution plot:", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col_choice], kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns available for distribution plots.")

    # Categorical column counts
    st.subheader("🧾 Value Counts for Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        cat_choice = st.selectbox("Select column for value count plot:", categorical_cols)
        fig, ax = plt.subplots()
        sns.countplot(x=cat_choice, data=df, palette="Set2", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No categorical columns available for count plots.")
