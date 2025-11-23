import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress
from typing import Dict, List
from typing import Optional


PLOTLY_CONFIG = {"displaylogo": False, "scrollZoom": True}

def _insight_box(lines: List[str]) -> None:
    clean_lines = [ln for ln in lines if ln]
    if not clean_lines:
        return
    st.markdown("####  Insight")
    for ln in clean_lines:
        st.markdown(f"- {ln}")

def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def _missing_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if col not in df.columns]


def render_payment_sankey(df: pd.DataFrame) -> None:
    st.subheader("Sankey – Payment Method → Category → Subscription Status")
    st.caption("Hover nodes to see totals, drag to reorder, and follow highlighted paths to compare funnels.")

    if df.empty:
        st.error("Dataset is empty.")
        return


    max_rows = len(df)
    default_rows = min(1000, max_rows)
    sample_rows = st.slider(
        "Maximum number of rows used for this Sankey",
        min_value=1,
        max_value=max_rows,
        value=default_rows,
        step=1,
        help="Reducing rows makes the chart faster and less cluttered.",
    )
    if sample_rows < max_rows:
        df = df.sample(sample_rows, random_state=42)

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
    try:
        total_flow = float(grouped["value"].sum())
        first_dim = path_columns[0]
        last_dim = path_columns[-1]

        by_first = grouped.groupby(first_dim)["value"].sum().sort_values(ascending=False)
        by_last = grouped.groupby(last_dim)["value"].sum().sort_values(ascending=False)

        top_first = by_first.index[0]
        top_first_val = int(by_first.iloc[0])

        top_last = by_last.index[0]
        top_last_val = int(by_last.iloc[0])

        top_path = grouped.sort_values("value", ascending=False).iloc[0]
        strongest_path_desc = " → ".join(str(top_path[c]) for c in path_columns)
        strongest_path_val = int(top_path["value"])

        _insight_box([
            f"Total flow across all paths: **{int(total_flow):,} {value_label.lower()}**.",
            f"Most common **{first_dim}** in the funnel: **{top_first}** ({top_first_val:,} {value_label.lower()}).",
            f"Most frequent **{last_dim}** at the end of the funnel: **{top_last}** ({top_last_val:,} {value_label.lower()}).",
            f"Strongest single path: **{strongest_path_desc}** "
            f"({strongest_path_val:,} {value_label.lower()}).",
        ])
    except Exception:
        # Insight hesaplanamazsa sessizce geç
        pass

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

    if df.empty:
        st.error("Dataset is empty.")
        return

    # 🔹 VERİYİ AZALTMA SLIDER'I
    max_rows = len(df)
    default_rows = min(2000, max_rows)
    sample_rows = st.slider(
        "Maximum number of rows used for this hierarchy chart",
        min_value=1,
        max_value=max_rows,
        value=default_rows,
        step=1,
        help="Use fewer rows to keep the chart fast and readable.",
    )
    if sample_rows < max_rows:
        df = df.sample(sample_rows, random_state=42)

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
    try:
        if "Category" in grouped.columns:
            by_cat = grouped.groupby("Category")["value"].sum().sort_values(ascending=False)
            top_cat = by_cat.index[0]
            top_cat_val = int(by_cat.iloc[0])

            unique_items = grouped["Item Purchased"].nunique() if "Item Purchased" in grouped.columns else None

            lines = [
                f"Dominant category in the hierarchy: **{top_cat}** "
                f"({top_cat_val:,} total {agg_mode})."
            ]
            if unique_items:
                lines.append(f"Total distinct items represented: **{unique_items:,}**.")
            _insight_box(lines)
    except Exception:
        pass
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)



def render_payment_category_network(df: pd.DataFrame) -> None:

    st.subheader("Network – Payment Method ↔ Category")
    st.caption(
        "Each node is a payment method or product category; edges show how often they co-occur. "
        "Thicker edges = stronger relationship."
    )

    # Kolon kontrolü
    needed_cols = ["Payment Method", "Category"]
    missing = _missing_columns(df, needed_cols)
    if missing:
        st.error(f"Dataset is missing: {', '.join(missing)}")
        return

    if df.empty:
        st.error("Dataset is empty.")
        return


    max_rows = len(df)
    default_rows = min(3000, max_rows)
    sample_rows = st.slider(
        "Number of rows used for the network",
        min_value=200,
        max_value=max_rows,
        value=default_rows,
        step=100,
        help="Fewer rows → fewer edges → cleaner network.",
    )
    if sample_rows < max_rows:
        df = df.sample(sample_rows, random_state=42)

    # Payment Method ↔ Category eşleşme sıklıkları
    grouped = (
        df.groupby(["Payment Method", "Category"])
        .size()
        .reset_index(name="weight")
        .sort_values("weight", ascending=False)
    )

    if grouped.empty:
        st.warning("No valid Payment Method ↔ Category pairs found.")
        return

    # Çok küçük edge’leri gizlemek için threshold slider
    max_weight = int(grouped["weight"].max())
    min_weight = st.slider(
        "Hide connections with weight below",
        min_value=1,
        max_value=max_weight,
        value=1,
        step=1,
    )
    grouped = grouped[grouped["weight"] >= min_weight]
    if grouped.empty:
        st.warning("No edges left after applying the threshold. Lower the slider.")
        return
    try:
        by_pm = grouped.groupby("Payment Method")["weight"].sum().sort_values(ascending=False)
        by_cat = grouped.groupby("Category")["weight"].sum().sort_values(ascending=False)

        top_pm = by_pm.index[0]
        top_pm_val = int(by_pm.iloc[0])

        top_cat = by_cat.index[0]
        top_cat_val = int(by_cat.iloc[0])

        strongest = grouped.sort_values("weight", ascending=False).iloc[0]
        strongest_pm = str(strongest["Payment Method"])
        strongest_cat = str(strongest["Category"])
        strongest_val = int(strongest["weight"])

        _insight_box([
            f"Most connected payment method: **{top_pm}** ({top_pm_val:,} purchases).",
            f"Most frequently purchased category overall: **{top_cat}** ({top_cat_val:,} purchases).",
            f"Strongest single link: **{strongest_pm} → {strongest_cat}** "
            f"({strongest_val:,} co-occurrences).",
        ])
    except Exception:
        pass
    # Node setleri
    payment_nodes = grouped["Payment Method"].astype(str).unique().tolist()
    category_nodes = grouped["Category"].astype(str).unique().tolist()

    # Node pozisyonları (iki sütunlu layout: solda payment, sağda category)
    x_payment = 0.1
    x_category = 0.9

    y_payment = np.linspace(0, 1, len(payment_nodes)) if payment_nodes else []
    y_category = np.linspace(0, 1, len(category_nodes)) if category_nodes else []

    # Node dict: isim → (x, y)
    node_positions: Dict[str, Dict[str, float]] = {}

    for name, y in zip(payment_nodes, y_payment):
        node_positions[f"pm::{name}"] = {"x": x_payment, "y": y}

    for name, y in zip(category_nodes, y_category):
        node_positions[f"cat::{name}"] = {"x": x_category, "y": y}

    # Edge traces (her edge için ayrı trace, kalınlık weight'e göre)
    edge_traces = []
    max_w = grouped["weight"].max()
    min_w = grouped["weight"].min()

    # Normalizasyon için küçük bir epsilon
    w_range = max(max_w - min_w, 1)

    for _, row in grouped.iterrows():
        pm = str(row["Payment Method"])
        cat = str(row["Category"])
        w = float(row["weight"])

        src = node_positions.get(f"pm::{pm}")
        tgt = node_positions.get(f"cat::{cat}")
        if src is None or tgt is None:
            continue

        # Weight'e göre çizgi kalınlığı (2–10 arası)
        width = 2 + 8 * ((w - min_w) / w_range)

        edge_traces.append(
            go.Scatter(
                x=[src["x"], tgt["x"]],
                y=[src["y"], tgt["y"]],
                mode="lines",
                line=dict(width=width, color="rgba(37,99,235,0.4)"),
                hoverinfo="text",
                text=f"{pm} → {cat}<br>Count: {int(w)}",
                showlegend=False,
            )
        )

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    # Payment method nodes
    for name, y in zip(payment_nodes, y_payment):
        node_x.append(x_payment)
        node_y.append(y)
        node_text.append(f"{name} (Payment)")
        node_color.append("#1d3557")

    # Category nodes
    for name, y in zip(category_nodes, y_category):
        node_x.append(x_category)
        node_y.append(y)
        node_text.append(f"{name} (Category)")
        node_color.append("#e76f51")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle right",
        marker=dict(size=14, color=node_color, line=dict(width=1, color="white")),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure()

    # Önce edge’ler, sonra node’lar (node’lar üstte kalsın)
    for et in edge_traces:
        fig.add_trace(et)

    fig.add_trace(node_trace)

    fig.update_layout(
        title="Payment Method ↔ Category Network",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(t=40, l=20, r=20, b=20),
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

US_STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
    "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}

US_STATE_CODES = set(US_STATE_ABBR.values())


def _guess_state_column(df: pd.DataFrame) -> Optional[str]:
    """Datasetteki olası eyalet kolonunu tahmin etmeye çalışır."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return None

    # Önce "State" veya benzeri isimlere bak
    for name in cat_cols:
        if name.lower() in ["state", "state_code", "state_name"]:
            return name

    # Değerleri eyalet ismi / kodu olan kolonları ara
    for col in cat_cols:
        vals = df[col].dropna().astype(str).unique().tolist()
        # Birkaç örneğe bakmak yeterli
        sample = vals[:50]
        for v in sample:
            v_strip = v.strip()
            if v_strip in US_STATE_ABBR or v_strip in US_STATE_CODES:
                return col

    return None


def _to_state_code(value: str) -> Optional[str]:
    """Full isim veya kodu 2 harfli state code'a çevir."""
    if not isinstance(value, str):
        value = str(value)
    v = value.strip()
    if v in US_STATE_CODES:  # Zaten kod
        return v
    if v in US_STATE_ABBR:   # Full isim
        return US_STATE_ABBR[v]
    return None


def render_us_category_map(df: pd.DataFrame) -> None:
    st.subheader("USA Map – Category Behaviour by State")
    st.caption(
        "Bu harita üç farklı modda çalışır: "
        "1) Her eyalet için en popüler kategori, "
        "2) Metriğe göre ısı haritası, "
        "3) Hover üzerinde kategori dağılımı."
    )

    if df.empty:
        st.error("Dataset is empty.")
        return

    # Eyalet kolonunu bul
    state_col = _guess_state_column(df)
    if state_col is None:
        st.error(
            "Harita için eyalet bilgisini bulamadım. Lütfen dataset'te US state bilgisi içeren bir kolon "
            "(örneğin 'State' veya 'State Code') olduğundan emin olun."
        )
        return

    # Kategorik kolonları bul
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.error("Kategori için kullanılabilecek kategorik kolon bulunamadı.")
        return

    # Varsayılan category col: 'Category' varsa onu seç
    default_cat_idx = 0
    for i, c in enumerate(cat_cols):
        if c.lower() == "category":
            default_cat_idx = i
            break

    col1, col2 = st.columns(2)
    with col1:
        category_col = st.selectbox("Category column", options=cat_cols, index=default_cat_idx)
    with col2:
        metric_candidates = [col for col in ["Purchase Amount (USD)", "Previous Purchases"] if col in df.columns]
        metric_options = ["count"] + metric_candidates if metric_candidates else ["count"]
        metric_choice = st.selectbox("Metric", options=metric_options, index=0)

    # Mod seçimi
    mode = st.selectbox(
        "Map mode",
        options=[
            "Most popular category per state",
            "Metric choropleth by state",
            "Metric choropleth + category breakdown on hover",
        ],
        index=0,
    )

    working = df.copy()

    # Satır sayısı
    max_rows = len(working)
    if max_rows == 0:
        st.error("No data left for mapping.")
        return

    min_slider = 100 if max_rows >= 100 else 1
    default_rows = min(3000, max_rows)
    sample_rows = st.slider(
        "Number of rows used for map aggregation",
        min_value=min_slider,
        max_value=max_rows,
        value=default_rows,
        step=100 if max_rows >= 1000 else 50,
        help="Daha az satır, daha hızlı ve sade bir harita anlamına gelir.",
    )
    if sample_rows < max_rows:
        working = working.sample(sample_rows, random_state=42)

    # State code'a çevir
    working["state_code"] = working[state_col].apply(_to_state_code)
    working = working.dropna(subset=["state_code"])
    if working.empty:
        st.error("Eyalet değerlerini US state koduna dönüştüremedim. Lütfen kolon formatını kontrol edin.")
        return

    # Ortak: state + category + metric tabanı
    if metric_choice == "count":
        base = (
            working.groupby(["state_code", category_col])
            .size()
            .reset_index(name="value")
        )
        value_label = "Count"
    else:
        if metric_choice not in working.columns:
            st.error(f"Metric column '{metric_choice}' bulunamadı.")
            return
        base = (
            working.groupby(["state_code", category_col])[metric_choice]
            .sum()
            .reset_index(name="value")
        )
        value_label = f"{metric_choice} (sum)"

    if base.empty:
        st.error("Aggregated data is empty.")
        return

    # ----------------------------
    # MODE 1:
    # ----------------------------
    if mode == "Most popular category per state":
        # Her eyalet için en yüksek value'ya sahip kategoriyi bul
        idx = base.groupby("state_code")["value"].idxmax()
        top_df = base.loc[idx].copy()
        top_df.rename(columns={category_col: "top_category"}, inplace=True)

        try:
            cat_counts = top_df["top_category"].value_counts()
            top_global_cat = cat_counts.index[0]
            top_global_cat_states = int(cat_counts.iloc[0])

            _insight_box([
                f"Most dominant category across states: **{top_global_cat}** "
                f"(top category in **{top_global_cat_states}** states).",
                f"Total states represented: **{top_df['state_code'].nunique()}**.",
            ])
        except Exception:
            pass

        fig = px.choropleth(
            top_df,
            locations="state_code",
            locationmode="USA-states",
            color="top_category",  # kategoriye göre renk
            scope="usa",
            hover_data={
                "state_code": False,
                "top_category": True,
                "value": True,
            },
            labels={"top_category": "Category", "value": value_label},
            title="Most Popular Category per State",
        )
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        return

    # ----------------------------
    # MODE 2 & 3
    # ----------------------------

    state_totals = base.groupby("state_code")["value"].sum().reset_index(name="total_value")
    try:
        top_row = state_totals.sort_values("total_value", ascending=False).iloc[0]
        _insight_box([
            f"State with highest {value_label.lower()}: **{top_row['state_code']}** "
            f"({int(top_row['total_value']):,}).",
            f"Number of states with non-zero {value_label.lower()}: "
            f"**{(state_totals['total_value'] > 0).sum()}**.",
        ])
    except Exception:
        pass

    if mode == "Metric choropleth by state":
        fig = px.choropleth(
            state_totals,
            locations="state_code",
            locationmode="USA-states",
            color="total_value",
            color_continuous_scale="Blues",
            scope="usa",
            labels={"total_value": value_label},
            title="Metric Choropleth by State",
        )
        fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        return

    # ----------------------------
    # MODE 3
    # ----------------------------
    rows = []
    for state in state_totals["state_code"].unique():
        sub = base[base["state_code"] == state].sort_values("value", ascending=False)
        if sub.empty:
            continue
        total = sub["value"].sum()
        top_cat = sub.iloc[0][category_col]

        # Çok uzun olmasın diye ilk 5 kategoriyi gösterelim
        breakdown_lines = []
        for _, r in sub.head(5).iterrows():
            breakdown_lines.append(f"{r[category_col]}: {int(r['value'])}")
        breakdown_text = "<br>".join(breakdown_lines)

        rows.append(
            {
                "state_code": state,
                "total_value": total,
                "top_category": top_cat,
                "breakdown": breakdown_text,
            }
        )

    breakdown_df = pd.DataFrame(rows)
    if breakdown_df.empty:
        st.error("No aggregated breakdown data to show.")
        return
    try:
        top_row = breakdown_df.sort_values("total_value", ascending=False).iloc[0]
        _insight_box([
            f"State with highest {value_label.lower()}: **{top_row['state_code']}** "
            f"({int(top_row['total_value']):,}).",
            f"Top category in that state: **{top_row['top_category']}**.",
            f"States represented in breakdown: **{breakdown_df['state_code'].nunique()}**.",
        ])
    except Exception:
        pass
    fig = px.choropleth(
        breakdown_df,
        locations="state_code",
        locationmode="USA-states",
        color="total_value",
        color_continuous_scale="Purples",
        scope="usa",
        labels={"total_value": value_label},
        title="Metric Choropleth with Category Breakdown on Hover",
        hover_data={"top_category": True, "breakdown": True, "total_value": True, "state_code": False},
    )

    # Custom hovertemplate, breakdown'ı düzgün gösterelim
    fig.update_traces(
        hovertemplate=(
            "Top category: %{customdata[0]}<br>"
            + value_label
            + ": %{z:,.0f}<br><br>"
            + "Category breakdown:<br>%{customdata[1]}<extra></extra>"
        )
    )

    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

