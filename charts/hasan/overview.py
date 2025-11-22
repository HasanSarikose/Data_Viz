import streamlit as st
import pandas as pd


def render_overview(df: pd.DataFrame):
    st.header("📌 Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("İlk 5 Satır")
        st.dataframe(df.head())

    with col2:
        st.subheader("Sayısal Kolon Özeti")
        st.dataframe(df.describe(include="all").T)

    st.markdown("---")

    st.subheader("🧱 Kolon Bilgileri")
    info_df = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "n_unique": [df[c].nunique() for c in df.columns],
        "n_missing": [df[c].isna().sum() for c in df.columns],
    })
    st.dataframe(info_df)
