import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


def _numeric_columns(df: pd.DataFrame):
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


STATE_ABBREVIATIONS: Dict[str, str] = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District Of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}
STATE_NAMES: Dict[str, str] = {abbr: name for name, abbr in STATE_ABBREVIATIONS.items()}
VALID_STATE_CODES = set(STATE_NAMES.keys())


def _state_to_code(value: str) -> str:
    if value is None or value == "":
        return ""
    cleaned = str(value).strip()
    if not cleaned:
        return ""
    upper = cleaned.upper()
    if len(upper) == 2 and upper in VALID_STATE_CODES:
        return upper
    normalized = cleaned.title()
    return STATE_ABBREVIATIONS.get(normalized, "")


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


def render_geographic_choropleth(df: pd.DataFrame) -> None:
    st.subheader("USA Choropleth – Purchase Heatmap by Location")
    st.caption("Zoom, pan, and hover to inspect state-level totals straight from the Location column.")

    if "Location" not in df.columns:
        st.error("Dataset must include a 'Location' column with US states.")
        return

    metric_candidates = [
        col for col in ["Purchase Amount (USD)", "Previous Purchases", "Review Rating"] if col in df.columns
    ]
    options = metric_candidates + ["count"] if metric_candidates else ["count"]
    metric_choice = st.selectbox("Heatmap metric", options=options, index=0)
    agg_choice = "sum"
    if metric_choice != "count":
        agg_choice = st.selectbox("Aggregation", options=["sum", "mean"], index=0)

    working_cols = ["Location"] + ([metric_choice] if metric_choice != "count" else [])
    working = df[working_cols].dropna(subset=["Location"]).copy()
    if working.empty:
        st.error("No rows with valid Location values.")
        return

    working["state_code"] = working["Location"].apply(_state_to_code)
    unknown_count = (working["state_code"] == "").sum()
    working = working[working["state_code"] != ""]
    if working.empty:
        st.error("Could not match any Location values to US states. Please check the column content.")
        return
    if unknown_count:
        st.warning(f"{unknown_count} rows skipped because their state names were not recognized.")

    if metric_choice == "count":
        grouped = working.groupby(["state_code"]).size().reset_index(name="value")
        value_label = "Count"
    else:
        grouped = (
            working.groupby(["state_code"])[metric_choice]
            .agg(agg_choice)
            .reset_index(name="value")
        )
        value_label = f"{metric_choice} ({agg_choice})"

    grouped["state_label"] = grouped["state_code"].map(STATE_NAMES)

    fig = px.choropleth(
        grouped,
        locations="state_code",
        color="value",
        locationmode="USA-states",
        color_continuous_scale="YlOrRd",
        scope="usa",
        hover_name="state_label",
        hover_data={"value": ":,.0f"},
    )
    fig.update_layout(
        margin=dict(t=10, l=10, r=10, b=10),
        coloraxis_colorbar=dict(title=value_label),
    )
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
