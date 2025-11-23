import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde
from typing import List, Tuple


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


PARALLEL_COLUMNS: List[Tuple[str, str]] = [
    ("Age", "Age"),
    ("Purchase Amount (USD)", "Purchase Amount"),
    ("Review Rating", "Review Rating"),
    ("Previous Purchases", "Previous Purchases"),
]


def _insight_box(title: str, lines: List[str]) -> None:
    clean_lines = [ln for ln in lines if ln]
    if not clean_lines:
        return
    st.markdown(f"**{title}**")
    for ln in clean_lines:
        st.markdown(f"- {ln}")


def _missing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _season_order(values: pd.Series) -> pd.Series:
    base_order = ["Winter", "Spring", "Summer", "Autumn"]
    mapping = {name: idx for idx, name in enumerate(base_order)}
    normalized = values.astype(str).str.strip().str.title().replace({"Fall": "Autumn"})
    order = normalized.map(mapping)
    if order.isna().any():
        extras = normalized[order.isna()].unique()
        extra_mapping = {
            name: len(mapping) + idx for idx, name in enumerate(sorted(extras))
        }
        order = order.fillna(normalized.map(extra_mapping))
    order = order.fillna(len(mapping))
    return order.astype(int), normalized


def render_parallel_coordinates(df: pd.DataFrame) -> None:
    st.subheader("Parallel Coordinates – Demographic Spend Profile")
    st.caption(
        "Drag on any axis to brush ranges, hover for tooltips, and compare who converts "
        "into high spenders."
    )

    required = [col for col, _ in PARALLEL_COLUMNS]
    missing = _missing_columns(df, required)
    if missing:
        st.error(f"Dataset is missing: {', '.join(missing)}")
        return

    working = df.dropna(subset=required).copy()
    if working.empty:
        st.error("No rows left after dropping records with missing numeric values.")
        return

    age_min, age_max = int(working["Age"].min()), int(working["Age"].max())
    purchase_min = float(working["Purchase Amount (USD)"].min())
    purchase_max = float(working["Purchase Amount (USD)"].max())
    if age_min == age_max or purchase_min == purchase_max:
        st.error("Need variability in Age and Purchase Amount to draw parallel coordinates.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        age_range = st.slider(
            "Age band",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
        )
    with col2:
        purchase_range = st.slider(
            "Purchase amount band",
            min_value=float(purchase_min),
            max_value=float(purchase_max),
            value=(purchase_min, purchase_max),
            step=1.0,
        )
    with col3:
        color_metric = st.selectbox(
            "Color by metric",
            options=[label for _, label in PARALLEL_COLUMNS],
            index=1,
        )
    metric_lookup = {label: col for col, label in PARALLEL_COLUMNS}

    working = working[
        working["Age"].between(*age_range)
        & working["Purchase Amount (USD)"].between(*purchase_range)
    ]
    if "Gender" in df.columns:
        genders = sorted(working["Gender"].dropna().unique().tolist())
        if genders:
            selected_gender = st.multiselect(
                "Focus gender(s)", options=genders, default=genders
            )
            working = working[working["Gender"].isin(selected_gender)]

    if working.empty:
        st.warning("No rows after applying the filters above.")
        return

    dimension_columns = [column for column, _ in PARALLEL_COLUMNS]
    label_lookup = {column: label for column, label in PARALLEL_COLUMNS}

    fig = px.parallel_coordinates(
        working,
        dimensions=dimension_columns,
        color=metric_lookup[color_metric],
        color_continuous_scale=px.colors.sequential.Viridis,
        labels=label_lookup,
    )
    fig.update_layout(
        coloraxis_colorbar=dict(title=color_metric),
        margin=dict(t=20, l=0, r=0, b=0),
    )
    _insight_box(
        "Parallel Coordinates Insight",
        [
            "Brush age and spend axes to isolate loyal cohorts; dense bundles show which segments convert together.",
            "Color encoding mirrors the chosen KPI, so sudden shifts in hue flag metric anomalies immediately.",
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_histogram_kde(df: pd.DataFrame) -> None:
    st.subheader("Histogram + KDE Overlay")
    st.caption("High-quality distribution insight with smoothing.")

    numeric_cols = _numeric_columns(df)
    if not numeric_cols:
        st.error("Need at least one numeric column.")
        return

    col = st.selectbox("Numeric column", options=numeric_cols, index=0)
    metric_name = col
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
    _insight_box(
        "Histogram + KDE Insight",
        [
            f"The long tail in {metric_name} exposes which categories produce extreme shoppers.",
            "The KDE overlay smooths the bars, making multi-modal bumps obvious without losing detail.",
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_season_line_slider(df: pd.DataFrame) -> None:
    st.subheader("Season vs Avg Purchase Amount + Age Slider")
    st.caption("Filter by age range with the controls and use the in-chart range slider to zoom specific seasons.")

    required = ["Season", "Purchase Amount (USD)", "Age"]
    missing = _missing_columns(df, required)
    if missing:
        st.error(f"Dataset is missing: {', '.join(missing)}")
        return

    working = df[required].dropna()
    if working.empty:
        st.error("No rows contain all Season, Age, and Purchase Amount values.")
        return

    age_min, age_max = int(working["Age"].min()), int(working["Age"].max())
    if age_min == age_max:
        st.error("Need more than one age value to build the slider.")
        return

    age_range = st.slider(
        "Filter by age range",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
    )
    filtered = working[working["Age"].between(*age_range)]
    if filtered.empty:
        st.warning("No rows remain for the selected age range.")
        return

    grouped = (
        filtered.groupby("Season")["Purchase Amount (USD)"]
        .mean()
        .reset_index(name="avg_purchase")
    )
    if grouped.empty:
        st.error("Aggregation failed because all Season values were filtered out.")
        return

    grouped["season_index"], grouped["season_label"] = _season_order(grouped["Season"])
    grouped = grouped.sort_values("season_index")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped["season_index"],
            y=grouped["avg_purchase"],
            text=grouped["season_label"],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=10, line=dict(width=1, color="white")),
            hovertemplate="Season: %{text}<br>Avg Purchase: $%{y:,.0f}<extra></extra>",
            name="Avg Purchase",
        )
    )
    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        xaxis=dict(
            title="Season",
            tickmode="array",
            tickvals=grouped["season_index"],
            ticktext=grouped["season_label"],
            rangeslider=dict(visible=True, thickness=0.08),
        ),
        yaxis=dict(title="Average Purchase Amount (USD)"),
        hovermode="x unified",
    )
    _insight_box(
        "Season Line + Slider Insight",
        [
            f"Age filter is pinned to {age_range[0]}–{age_range[1]} so we can narrate exactly when demand peaks.",
            "Use the range slider to zoom into Winter→Autumn blocks and explain how each age band reacts.",
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Age range applied: {age_range[0]} - {age_range[1]} years.")
