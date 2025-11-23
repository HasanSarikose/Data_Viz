import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde, linregress
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


def _numeric_columns(df: pd.DataFrame):
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def render_kmeans_clusters(df: pd.DataFrame) -> None:
    st.subheader("K-Means Clustering – Age vs Purchase Amount")
    st.caption(
        "3-cluster segmentation built with Age, Purchase Amount, and Previous Purchases. "
        "Hover to inspect customers, lasso to focus, and use the multiselect to highlight clusters."
    )

    required = ["Age", "Purchase Amount (USD)", "Previous Purchases"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Dataset is missing: {', '.join(missing)}")
        return

    working = df[required].dropna().copy()
    if len(working) < 3:
        st.error("Need at least three complete observations to run clustering.")
        return

    scaler = StandardScaler()
    features = working[required]
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    working["cluster_id"] = kmeans.fit_predict(scaled)
    working["Cluster"] = working["cluster_id"].apply(lambda idx: f"Cluster {idx + 1}")

    centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=required,
    )
    centers["Cluster"] = [f"Cluster {idx + 1}" for idx in range(len(centers))]

    cluster_options = sorted(working["Cluster"].unique())
    selected = st.multiselect(
        "Clusters to display",
        options=cluster_options,
        default=cluster_options,
    )
    plot_df = working if not selected else working[working["Cluster"].isin(selected)]
    center_plot = centers if not selected else centers[centers["Cluster"].isin(selected)]

    if plot_df.empty:
        st.warning("No clusters selected. Choose at least one cluster to visualize.")
        return

    fig = px.scatter(
        plot_df,
        x="Age",
        y="Purchase Amount (USD)",
        color="Cluster",
        size="Previous Purchases",
        hover_data={"Previous Purchases": True},
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white")))

    if not center_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=center_plot["Age"],
                y=center_plot["Purchase Amount (USD)"],
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=16,
                    color="#0f172a",
                    line=dict(width=2, color="white"),
                ),
                name="Centroid",
                hovertemplate=(
                    "Centroid %{text}<br>"
                    "Age: %{x:.1f}<br>"
                    "Purchase: $%{y:,.0f}<br>"
                    "Prev Purchases: %{customdata:.1f}<extra></extra>"
                ),
                text=center_plot["Cluster"],
                customdata=center_plot["Previous Purchases"],
            )
        )

    fig.update_layout(
        margin=dict(t=20, l=0, r=0, b=0),
        legend_title="Cluster",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    summary = centers.rename(
        columns={
            "Age": "Age (avg)",
            "Purchase Amount (USD)": "Purchase (avg)",
            "Previous Purchases": "Prev Purchases (avg)",
        }
    )
    summary = summary[["Cluster", "Age (avg)", "Purchase (avg)", "Prev Purchases (avg)"]]
    st.dataframe(summary.round(1), use_container_width=True)


def render_histogram_kde(df: pd.DataFrame) -> None:
    st.subheader("Histogram + KDE Overlay")
    st.caption("High-quality distribution insight with smoothing.")

    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.error("Need at least one numeric column.")
        return

    col = st.selectbox("Numeric column", options=numeric_cols, index=0)
    data = df[col].dropna()
    if data.empty:
        st.error("No numeric values to plot.")
        return

    bins = st.slider("Number of bins", min_value=10, max_value=60, value=30, step=5)
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    ys = kde(xs)
    ys_scaled = ys * (data.max() - data.min()) * len(data) / bins

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=data,
            nbinsx=bins,
            name="Histogram",
            marker_color="#8ecae6",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys_scaled,
            name="KDE",
            line=dict(color="#023047", width=3),
            yaxis="y",
        )
    )
    fig.update_layout(
        bargap=0.05,
        margin=dict(t=20, l=0, r=0, b=0),
        yaxis_title="Count / Density",
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_scatter_regression(df: pd.DataFrame) -> None:
    st.subheader("Scatter with Regression Line")
    st.caption("Seaborn-style trendline for the statistical speaker.")

    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns.")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_col = st.selectbox("X axis", options=numeric_cols, index=0)
    with col2:
        y_options = [col for col in numeric_cols if col != x_col]
        if not y_options:
            st.error("Need different columns for X and Y.")
            return
        y_col = st.selectbox("Y axis", options=y_options, index=0)

    subset = df[[x_col, y_col]].dropna()
    if subset.empty:
        st.error("No overlapping values to regress.")
        return

    slope, intercept, r_value, _, _ = linregress(subset[x_col], subset[y_col])
    line_x = np.linspace(subset[x_col].min(), subset[x_col].max(), 100)
    line_y = intercept + slope * line_x

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            marker=dict(size=8, color="#219ebc", opacity=0.7),
            name="Data points",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="#fb8500", width=3),
            name=f"Regression (R^2={r_value**2:.2f})",
        )
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        margin=dict(t=20, l=0, r=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_box_plot(df: pd.DataFrame) -> None:
    st.subheader("Box Plot (Standard)")
    st.caption("Simple distribution comparison by category.")

    numeric_cols = _numeric_columns(df)
    categorical_cols = [col for col in df.columns if df[col].dtype == object]
    if not numeric_cols or not categorical_cols:
        st.error("Need at least one numeric and one categorical column.")
        return

    col1, col2 = st.columns(2)
    with col1:
        metric_col = st.selectbox("Numeric column", options=numeric_cols, index=0)
    with col2:
        category_col = st.selectbox("Category column", options=categorical_cols, index=0)

    fig = px.box(df, x=category_col, y=metric_col, color=category_col, points="outliers")
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=metric_col,
        showlegend=False,
        margin=dict(t=20, l=0, r=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
