import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}


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


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


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
    metric_choice = st.selectbox("Flow metric", options=metric_options, index=0 if metric_options[0] == "count" else 1)

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


def render_geographic_choropleth(df: pd.DataFrame) -> None:
    st.subheader("USA Choropleth – Purchase Heatmap by Location")
    st.caption("Zoom, pan, and hover to inspect state-level totals. Aggregates directly from the Location column.")

    if "Location" not in df.columns:
        st.error("Dataset must include a 'Location' column with US states.")
        return

    metric_candidates = [col for col in ["Purchase Amount (USD)", "Previous Purchases", "Review Rating"] if col in df.columns]
    metric_choice = st.selectbox(
        "Heatmap metric",
        options=metric_candidates + ["count"] if metric_candidates else ["count"],
        index=0,
    )
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


def render_season_line_slider(df: pd.DataFrame) -> None:
    st.subheader("Season vs Avg Purchase Amount + Age Slider")
    st.caption("Filter by age range with the controls and use the in-chart range slider to zoom specific seasons.")

    required = ["Season", "Purchase Amount (USD)", "Age"]
    missing = [col for col in required if col not in df.columns]
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
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    st.caption(f"Age range applied: {age_range[0]} - {age_range[1]} years.")
