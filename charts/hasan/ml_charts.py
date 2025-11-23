# charts/ml/ml_charts.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]


def _insight_box(lines: List[str]) -> None:
    clean = [ln for ln in lines if ln]
    if not clean:
        return
    st.markdown("#### 📌 ML Insight")
    for ln in clean:
        st.markdown(f"- {ln}")


def render_kmeans_clusters(df: pd.DataFrame) -> None:
    st.subheader("K-Means Customer Segmentation")
    st.caption(
        "Basic ML clustering: groups customers into segments based on numeric features. "
        "Use the controls to pick features and cluster count."
    )

    if df.empty:
        st.error("Dataset is empty.")
        return

    num_cols = _numeric_columns(df)
    if len(num_cols) < 2:
        st.error("K-Means clustering requires at least two numeric columns.")
        return

    # Özellik seçimi
    st.markdown("**Select features to cluster on**")
    default_feats = []
    for cand in ["Age", "Purchase Amount (USD)", "Previous Purchases", "Review Rating"]:
        if cand in num_cols:
            default_feats.append(cand)
    if len(default_feats) < 2:
        default_feats = num_cols[: min(3, len(num_cols))]

    selected_features = st.multiselect(
        "Features",
        options=num_cols,
        default=default_feats,
        help="These numeric columns will be used as input features for K-Means.",
    )
    if len(selected_features) < 2:
        st.warning("Select at least two features to run clustering.")
        return

    # Plot eksenleri
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X axis", options=selected_features, index=0)
    with col2:
        y_candidates = [c for c in selected_features if c != x_axis]
        if not y_candidates:
            st.error("X ve Y eksenleri için farklı sütunlar seçmelisin.")
            return
        y_axis = st.selectbox("Y axis", options=y_candidates, index=0)

    # K değeri
    k = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=3)

    # Satır sayısını sınırlama
    max_rows = len(df)
    default_rows = min(5000, max_rows)
    sample_rows = st.slider(
        "Number of rows used for clustering",
        min_value=min(500, max_rows),
        max_value=max_rows,
        value=default_rows,
        step=500 if max_rows > 1000 else 100,
        help="Fewer rows make clustering faster and the scatter plot clearer.",
    )
    if sample_rows < max_rows:
        work = df.sample(sample_rows, random_state=42).copy()
    else:
        work = df.copy()

    work_num = work[selected_features].dropna()
    if work_num.empty:
        st.error("No valid rows left after dropping NaNs for the selected features.")
        return

    # Standardizasyon + KMeans
    scaler = StandardScaler()
    X = scaler.fit_transform(work_num.values)

    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    clusters = model.fit_predict(X)

    work_num = work_num.copy()
    work_num["cluster"] = clusters

    # 2D scatter
    fig = px.scatter(
        work_num,
        x=x_axis,
        y=y_axis,
        color="cluster",
        color_continuous_scale="Turbo",
        title="K-Means Clusters",
        hover_data=selected_features,
    )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    # Basit cluster özeti
    try:
        cluster_sizes = work_num["cluster"].value_counts().sort_index()
        largest_cluster = int(cluster_sizes.idxmax())
        largest_size = int(cluster_sizes.max())

        # Eğer Purchase Amount varsa ortalamaları da göster
        amount_insight = ""
        if "Purchase Amount (USD)" in work_num.columns:
            means = (
                work_num.groupby("cluster")["Purchase Amount (USD)"]
                .mean()
                .round(0)
                .to_dict()
            )
            best_cluster = max(means, key=means.get)
            amount_insight = (
                f"Cluster **{best_cluster}** has the highest average purchase amount "
                f"(${means[best_cluster]:,.0f})."
            )

        _insight_box(
            [
                f"Model created **{k} clusters** based on {len(selected_features)} features.",
                f"Largest cluster is **#{largest_cluster}** with **{largest_size:,} customers**.",
                amount_insight,
            ]
        )
    except Exception:
        # Insight hesaplanamazsa sessiz geç
        pass
