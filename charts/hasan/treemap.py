import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def render_treemap(df: pd.DataFrame):
    st.header("🌳 Treemap (Dinamik Sütun Seçimi)")

    if df.empty or df.shape[1] == 0:
        st.error("Treemap için geçerli bir veri yok.")
        return

    # Sütun tiplerine göre ayır
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(cat_cols) == 0 or len(num_cols) == 0:
        st.error("Treemap için en az bir kategorik ve bir sayısal sütun gerekli.")
        return

    st.markdown("Önce hangi sütunları kullanacağını seç 👇")

    col1, col2, col3 = st.columns(3)

    with col1:
        path_col = st.selectbox(
            "📂 Kategorik sütun (Treemap blokları için)",
            options=cat_cols,
            index=0,
        )

    with col2:
        value_col = st.selectbox(
            "💰 Sayısal sütun (Blok alanı için)",
            options=num_cols,
            index=0,
        )

    with col3:
        agg_func = st.selectbox(
            "🔧 Aggregation",
            options=["mean", "sum", "count"],
            index=0,
        )

    # Color column seçimi (opsiyonel)
    color_option = st.selectbox(
        "🎨 Renk için sütun",
        options=["Aynı (value sütunu)"] + num_cols,
        index=0,
    )

    if color_option == "Aynı (value sütunu)":
        color_col = value_col
    else:
        color_col = color_option

    # Aggregation
    if agg_func == "mean":
        df_agg = df.groupby(path_col, dropna=False)[value_col].mean().reset_index()
    elif agg_func == "sum":
        df_agg = df.groupby(path_col, dropna=False)[value_col].sum().reset_index()
    else:  # count
        df_agg = df.groupby(path_col, dropna=False)[value_col].count().reset_index()

    df_agg = df_agg.sort_values(value_col, ascending=False)

    st.markdown(
        f"Seçilen yapı: **path = {path_col}**, **value = {value_col} ({agg_func})**, "
        f"**color = {color_col}**"
    )

    # Treemap çiz
    try:
        fig = px.treemap(
            df_agg,
            path=[path_col],
            values=value_col,
            color=color_col if color_col in df_agg.columns else value_col,
            color_continuous_scale="Tealgrn",
            title=f"Treemap – {path_col} vs {value_col} ({agg_func})",
        )
        fig.update_layout(margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Treemap çizilirken bir hata oluştu: {e}")
