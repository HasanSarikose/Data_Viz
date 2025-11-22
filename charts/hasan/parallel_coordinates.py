# charts/fng/parallel_coordinates.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def render_parallel_coordinates(df: pd.DataFrame):
    st.header("📐 Parallel Coordinates (Dynamic)")

    if df.empty:
        st.error("Geçerli bir veri bulunamadı.")
        return

    # Sadece sayısal kolonlar
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("Parallel coordinates için en az 2 sayısal sütun gerekli.")
        return

    st.markdown("Kullanmak istediğin sayısal sütunları seç:")

    selected_cols = st.multiselect(
        "📊 Numeric columns (2+)",
        num_cols,
        default=num_cols[:4] if len(num_cols) >= 4 else num_cols[:2],
    )

    if len(selected_cols) < 2:
        st.warning("En az iki sayısal sütun seçmelisin.")
        return

    color_col = st.selectbox(
        "🎨 Renk için kolon",
        options=selected_cols,
        index=0
    )

    st.markdown("---")

    # NaN'leri temizle (yoksa Plotly saçmalayabiliyor)
    df_plot = df[selected_cols].dropna()

    if df_plot.empty:
        st.error("Seçilen sütunlarda geçerli veri kalmadı (hepsi NaN olabilir).")
        return

    try:
        fig = px.parallel_coordinates(
            df_plot,
            dimensions=selected_cols,
            color=color_col,
            color_continuous_scale=px.colors.diverging.Tealrose,
            title="Parallel Coordinates Plot"
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Parallel coordinates çizilirken hata oluştu: {e}")
