import pandas as pd
import plotly.express as px
import streamlit as st
from typing import List, Tuple


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


PARALLEL_COLUMNS: List[Tuple[str, str]] = [
    ("Age", "Age"),
    ("Purchase Amount (USD)", "Purchase Amount"),
    ("Review Rating", "Review Rating"),
    ("Previous Purchases", "Previous Purchases"),
]


def _missing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


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


def render_interactive_box_plot(df: pd.DataFrame) -> None:
    st.subheader("Interactive Box Plot")
    st.caption("Dropdown selects the metric (Age / Purchase Amount / Review Rating) and compares by category.")

    metric_options = [
        ("Age", "Age"),
        ("Purchase Amount", "Purchase Amount (USD)"),
        ("Review Rating", "Review Rating"),
    ]
    available_metrics = [(label, col) for label, col in metric_options if col in df.columns]
    if not available_metrics:
        st.error("None of the required numeric columns are present.")
        return

    categorical_cols = [col for col in df.columns if df[col].dtype == object]
    if not categorical_cols:
        st.error("Need at least one categorical column to group by.")
        return

    metric_label = st.selectbox("Choose variable", options=[label for label, _ in available_metrics])
    metric_col = dict(available_metrics)[metric_label]

    category_index = categorical_cols.index("Category") if "Category" in categorical_cols else 0
    category_col = st.selectbox("Split by category", options=categorical_cols, index=category_index)

    point_options = {
        "All points": "all",
        "Suspected outliers": "suspectedoutliers",
        "Only outliers": "outliers",
        "Hide points": False,
    }
    show_points_label = st.selectbox("Point display", options=list(point_options.keys()), index=0)
    show_points = point_options[show_points_label]

    subset = df[[metric_col, category_col]].dropna()
    if subset.empty:
        st.warning("No rows available for the chosen configuration.")
        return

    median_order = (
        subset.groupby(category_col)[metric_col]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = px.box(
        subset,
        x=category_col,
        y=metric_col,
        color=category_col,
        points=show_points,
    )
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=metric_label,
        boxmode="group",
        showlegend=False,
        margin=dict(t=20, l=0, r=0, b=0),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=median_order)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
