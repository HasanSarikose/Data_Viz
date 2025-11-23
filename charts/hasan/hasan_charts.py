import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress
from typing import Dict, List


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _missing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def render_payment_sankey(df: pd.DataFrame) -> None:
    st.subheader("Sankey – Payment Method → Category → Subscription Status")
    st.caption("Hover nodes to see totals, drag to reorder, and follow highlighted paths to compare funnels.")

    path_columns = ["Payment Method", "Category", "Subscription Status"]
    missing = [col for col in path_columns if col not in df.columns]
    if missing:
        st.error(f"Dataset is missing: {', '.join(missing)}")
        return

    metric_candidates = [col for col in ["Purchase Amount (USD)", "Previous Purchases"] if col in df.columns]
    metric_options = ["count"] + metric_candidates if metric_candidates else ["count"]
    metric_choice = st.selectbox(
        "Flow metric",
        options=metric_options,
        index=0 if metric_options[0] == "count" else 1,
    )

    agg_choice = "sum"
    if metric_choice != "count":
        agg_choice = st.selectbox("Aggregation", options=["sum", "mean"], index=0)

    working_cols = path_columns + ([metric_choice] if metric_choice != "count" else [])
    working = df[working_cols].dropna(subset=path_columns)
    if working.empty:
        st.error("No rows left after removing records with missing path values.")
        return

    if metric_choice == "count":
        grouped = working.groupby(path_columns).size().reset_index(name="value")
        value_label = "Count"
    else:
        group = working.groupby(path_columns)[metric_choice]
        data = group.sum() if agg_choice == "sum" else group.mean()
        grouped = data.reset_index(name="value")
        value_label = f"{metric_choice} ({agg_choice})"

    flow_max = float(grouped["value"].max())
    min_flow = 0.0
    if flow_max > 0:
        step_value = max(flow_max / 200, 0.01)
        min_flow = st.slider(
            "Hide flows below",
            min_value=0.0,
            max_value=float(flow_max),
            value=0.0,
            step=float(step_value),
        )
        if min_flow > 0:
            grouped = grouped[grouped["value"] >= min_flow]
            if grouped.empty:
                st.warning("No links remain after applying the threshold. Lower the slider to see flows.")
                return

    labels: List[str] = []
    node_lookup: Dict[str, int] = {}
    for column in path_columns:
        for value in grouped[column].astype(str).unique():
            key = f"{column}::{value}"
            if key not in node_lookup:
                node_lookup[key] = len(labels)
                labels.append(f"{value} ({column})")

    sources: List[int] = []
    targets: List[int] = []
    values: List[float] = []

    for idx in range(len(path_columns) - 1):
        left, right = path_columns[idx], path_columns[idx + 1]
        pairs = grouped.groupby([left, right])["value"].sum().reset_index()
        for _, row in pairs.iterrows():
            source_key = f"{left}::{row[left]}"
            target_key = f"{right}::{row[right]}"
            sources.append(node_lookup[source_key])
            targets.append(node_lookup[target_key])
            values.append(float(row["value"]))

    if not values:
        st.warning("Flow values are zero across the board.")
        return

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=20,
                    thickness=18,
                    label=labels,
                    line=dict(color="#d9e3f0", width=1),
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(37,99,235,0.35)",
                    hovertemplate=f"%{{source.label}} → %{{target.label}}<br>{value_label}: %{{value:,.0f}}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=10),
        font=dict(size=12),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_sunburst_treemap(df: pd.DataFrame) -> None:
    st.subheader("Category → Item → Color (Sunburst / Treemap)")
    st.caption("Click to zoom down the hierarchy, hover for totals, collapse with double-click.")

    hierarchy = ["Category", "Item Purchased", "Color"]
    missing = _missing_columns(df, hierarchy)
    if missing:
        st.error(f"Dataset is missing: {', '.join(missing)}")
        return

    metric_candidates = [col for col in ["Purchase Amount (USD)", "Previous Purchases"] if col in df.columns]
    agg_metric = metric_candidates[0] if metric_candidates else None

    chart_type = st.selectbox("Chart type", options=["Sunburst", "Treemap"], index=0)
    agg_mode = st.selectbox("Aggregation", options=["sum", "mean", "count"], index=0)

    working = df[hierarchy + ([agg_metric] if agg_metric else [])].dropna(subset=hierarchy)
    if working.empty:
        st.error("No rows with the full Category → Item → Color combination.")
        return

    if agg_mode == "count" or agg_metric is None:
        grouped = working.groupby(hierarchy).size().reset_index(name="value")
        hover_template = "Count: %{value:,}"
        color_title = "Count"
    else:
        group = working.groupby(hierarchy)[agg_metric]
        data = group.sum() if agg_mode == "sum" else group.mean()
        grouped = data.reset_index(name="value")
        hover_template = f"{agg_metric} ({agg_mode}): %{{value:,.0f}}"
        color_title = f"{agg_metric} ({agg_mode})"

    if chart_type == "Sunburst":
        fig = px.sunburst(
            grouped,
            path=hierarchy,
            values="value",
            color="value",
            color_continuous_scale="Turbo",
        )
    else:
        fig = px.treemap(
            grouped,
            path=hierarchy,
            values="value",
            color="value",
            color_continuous_scale="Turbo",
        )

    fig.update_traces(hovertemplate=hover_template)
    fig.update_coloraxes(colorbar=dict(title=color_title))
    fig.update_layout(margin=dict(t=20, l=0, r=0, b=0))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_scatter_regression(df: pd.DataFrame) -> None:
    st.subheader("Scatter with Regression Line")
    st.caption("Seaborn-style trendline for Hasan'ın sahnesi.")

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
