import pandas as pd
import streamlit as st
from typing import Optional, Any

from utils.data_loader import (
    clean_data_drop_incomplete_rows,
    load_default_data,
    load_uploaded_data,
)

from charts.onder.onder_charts import (
    render_parallel_coordinates,
    render_histogram_kde,
    render_season_line_slider,
)
from charts.hasan.hasan_charts import (
    render_payment_sankey,
    render_sunburst_treemap,
    render_scatter_regression,
)
from charts.mustafa.mustafa_charts import (
    render_kmeans_clusters,
    render_geographic_choropleth,
    render_interactive_box_plot,
)


st.set_page_config(
    page_title="Shopping Behaviour Intelligence",
    layout="wide",
    page_icon=":bar_chart:",
)

CUSTOM_CSS = """
<style>
.hero-title {
    font-size: 2.3rem;
    font-weight: 700;
}
.hero-subtitle {
    color: #5f6368;
    font-size: 1rem;
}
div[data-testid="column"] div.stButton > button {
    height: 90px;
    border-radius: 18px;
    font-size: 1rem;
    font-weight: 600;
    border: 2px solid #d0d7de;
}
.chart-hint {
    color: #6b7280;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.chart-stage {
    animation: fadeSlide 0.8s ease;
    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(246,248,252,0.95));
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 15px 40px rgba(15,23,42,0.08);
}
@keyframes fadeSlide {
    from {
        opacity: 0;
        transform: translateY(12px) scale(0.98);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


HOME_KEY = "home"
HOME_LABEL = "Ana Sahne"
PRESENTERS = {
    "onder": {
        "label": "Husnu Onder Kabadayi ",
        "description": "Onder sahnesinde paralel koordinatlar, histogram + KDE ve sezon trendi yer alıyor.",
        "charts": [
            {
                "id": "parallel_coordinates",
                "label": "Parallel Coordinates",
                "quality": "Advanced",
                "func": render_parallel_coordinates,
            },
            {
                "id": "hist_kde",
                "label": "Histogram + KDE",
                "quality": "Advanced",
                "func": render_histogram_kde,
            },
            {
                "id": "season_line_slider",
                "label": "Season Line + Slider",
                "quality": "Medium",
                "func": render_season_line_slider,
            },
        ],
    },
    "hasan": {
        "label": "Hasan Sarikose ",
        "description": "Hasan sahnesi Sankey, kategori sunburst ve regresyon scatter ile devam ediyor.",
        "charts": [
            {
                "id": "payment_sankey",
                "label": "Payment Sankey",
                "quality": "Advanced",
                "func": render_payment_sankey,
            },
            {
                "id": "sunburst_treemap",
                "label": "Sunburst / Treemap",
                "quality": "Advanced",
                "func": render_sunburst_treemap,
            },
            {
                "id": "scatter_regression",
                "label": "Regression Scatter",
                "quality": "Medium",
                "func": render_scatter_regression,
            },
        ],
    },
    "mustafa": {
        "label": "Mustafa Sekeroglu ",
        "description": "Mustafa sahnesinde K-Means kümeleme, USA choropleth ve interaktif box plot var.",
        "charts": [
            {
                "id": "kmeans_clusters",
                "label": "K-Means Clusters",
                "quality": "Advanced",
                "func": render_kmeans_clusters,
            },
            {
                "id": "choropleth",
                "label": "USA Choropleth",
                "quality": "Advanced",
                "func": render_geographic_choropleth,
            },
            {
                "id": "interactive_box",
                "label": "Interactive Box Plot",
                "quality": "Advanced",
                "func": render_interactive_box_plot,
            },
        ],
    },
}



def _init_chart_state() -> None:
    if "active_charts" not in st.session_state:
        st.session_state["active_charts"] = {
            key: presenter["charts"][0]["id"] for key, presenter in PRESENTERS.items()
        }
    if "show_uploader" not in st.session_state:
        st.session_state["show_uploader"] = False


def render_hero() -> None:
    st.markdown('<div class="hero-title">Shopping Behaviour Control Room</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Upload fresh data from the navbar when needed and use the sidebar to pick who is presenting.</div>',
        unsafe_allow_html=True,
    )


def render_navbar() -> Optional[Any]:
    st.markdown("---")
    nav = st.container()
    with nav:
        cols = st.columns([0.7, 0.3])
        with cols[0]:
            st.markdown("**Active dataset:** shopping_behavior.csv (default)")
        with cols[1]:
            label = "Close uploader" if st.session_state["show_uploader"] else "Upload new CSV"
            if st.button(label, use_container_width=True):
                st.session_state["show_uploader"] = not st.session_state["show_uploader"]

    uploaded_file = None
    if st.session_state["show_uploader"]:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    return uploaded_file


def render_kpis(df: pd.DataFrame) -> None:
    if df.empty:
        return

    total_revenue = df["Purchase Amount (USD)"].sum() if "Purchase Amount (USD)" in df.columns else None
    avg_ticket = df["Purchase Amount (USD)"].mean() if "Purchase Amount (USD)" in df.columns else None
    unique_customers = df["Customer ID"].nunique() if "Customer ID" in df.columns else len(df)
    subscriber_rate = None
    if "Subscription Status" in df.columns:
        yes_mask = df["Subscription Status"].astype(str).str.strip().str.lower().eq("yes")
        if yes_mask.any():
            subscriber_rate = yes_mask.mean() * 100
    review_rating = df["Review Rating"].mean() if "Review Rating" in df.columns else None

    metrics = [
        ("Total revenue", f"${total_revenue:,.0f}" if total_revenue is not None else "NA"),
        ("Avg. ticket", f"${avg_ticket:,.0f}" if avg_ticket is not None else "NA"),
        ("Unique customers", f"{unique_customers:,}"),
        ("Subscriber rate", f"{subscriber_rate:.1f}%" if subscriber_rate is not None else "NA"),
        ("Avg. review", f"{review_rating:.2f}" if review_rating is not None else "NA"),
    ]
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def render_data_snapshot(df: pd.DataFrame) -> None:
    if df.empty:
        return
    with st.expander("Peek at the dataset"):
        st.dataframe(df.head(100))
        st.caption(f"{len(df):,} rows x {len(df.columns)} columns")


def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    filter_specs = [
        ("Gender", "Gender"),
        ("Category", "Category"),
        ("Season", "Season"),
        ("Payment method", "Payment Method"),
        ("Frequency", "Frequency of Purchases"),
        ("Subscription status", "Subscription Status"),
    ]
    filter_specs = [(label, column) for label, column in filter_specs if column in df.columns]
    if not filter_specs:
        return df

    st.markdown("### Shared filters")
    st.caption("Whatever you choose here will impact the three chart buttons below.")
    filtered = df.copy()
    filter_cols = st.columns(3)
    for idx, (label, column) in enumerate(filter_specs):
        options = sorted(filtered[column].dropna().unique().tolist())
        with filter_cols[idx % len(filter_cols)]:
            chosen = st.multiselect(label, options=options, placeholder="All")
        if chosen:
            filtered = filtered[filtered[column].isin(chosen)]

    st.success(f"{len(filtered):,} rows selected out of {len(df):,}.")
    return filtered


def _store_dataset(dataset: pd.DataFrame, info_msg: str, before: int, after: int) -> pd.DataFrame:
    st.session_state["dataset"] = dataset
    st.session_state["dataset_info"] = info_msg
    st.session_state["dataset_cleaning"] = f"Auto-clean removed {before - after} rows (kept {after})."
    return dataset


def load_dataset(uploaded_file: Optional[Any], show_messages: bool) -> pd.DataFrame:
    dataset = st.session_state.get("dataset")
    info_msg = st.session_state.get("dataset_info")
    cleaning_msg = st.session_state.get("dataset_cleaning")

    if uploaded_file is not None:
        raw = load_uploaded_data(uploaded_file)
        raw, before, after = clean_data_drop_incomplete_rows(raw)
        dataset = _store_dataset(raw, "Custom CSV uploaded from the navbar.", before, after)
        info_msg = st.session_state["dataset_info"]
        cleaning_msg = st.session_state["dataset_cleaning"]
    elif dataset is None:
        raw = load_default_data()
        raw, before, after = clean_data_drop_incomplete_rows(raw)
        dataset = _store_dataset(raw, "Using bundled shopping_behavior.csv.", before, after)
        info_msg = st.session_state["dataset_info"]
        cleaning_msg = st.session_state["dataset_cleaning"]

    if show_messages:
        if info_msg:
            st.caption(info_msg)
        if cleaning_msg:
            st.caption(cleaning_msg)

    return dataset.copy() if dataset is not None else pd.DataFrame()


def select_presenter() -> str:
    with st.sidebar:
        st.header("Presenters")
        options = [HOME_KEY] + list(PRESENTERS.keys())
        presenter_key = st.radio(
            "Choose who is on stage",
            options=options,
            format_func=lambda key: HOME_LABEL if key == HOME_KEY else PRESENTERS[key]["label"],
        )
    return presenter_key


def render_chart_switcher(presenter_key: str, df: pd.DataFrame) -> None:
    presenter = PRESENTERS[presenter_key]
    charts = presenter["charts"]
    chart_ids = [chart["id"] for chart in charts]
    active_id = st.session_state["active_charts"].get(presenter_key)
    if active_id not in chart_ids:
        active_id = chart_ids[0]
        st.session_state["active_charts"][presenter_key] = active_id

    columns = st.columns(len(charts))
    for idx, chart in enumerate(charts):
        label = chart["label"]
        if columns[idx].button(
            label,
            use_container_width=True,
            key=f"{presenter_key}_{chart['id']}",
        ):
            st.session_state["active_charts"][presenter_key] = chart["id"]
            active_id = chart["id"]

    active_chart = next((chart for chart in charts if chart["id"] == active_id), charts[0])

    chart_container = st.container()
    with chart_container:
        st.markdown('<div class="chart-stage">', unsafe_allow_html=True)
        active_chart["func"](df)
        st.markdown("</div>", unsafe_allow_html=True)


def render_home_stage(df: pd.DataFrame) -> None:
    st.markdown("### Ana Sahne – Dataset Özeti")
    st.markdown(
        """
        Shopping_behavior.csv; müşteri yaşları, alışveriş tutarı, sezon, kategori, abonelik ve ödeme tercihleri gibi
        alanları içerir. KPI kartları ve filtreler üzerinden istediğin kişi sahnesine geçmeden önce ihtiyacın olan
        tüm bilgiyi toparlayabilirsin.
        """
    )
    col_left, col_right = st.columns(2)
    with col_left:
        st.metric("Aktif satır sayısı", f"{len(df):,}")
    with col_right:
        st.metric("Sütun sayısı", f"{len(df.columns):,}")


def main() -> None:
    _init_chart_state()
    presenter_key = select_presenter()
    show_home = presenter_key == HOME_KEY

    uploaded_file = None
    if show_home:
        render_hero()
        uploaded_file = render_navbar()

    df = load_dataset(uploaded_file, show_messages=show_home)
    if df.empty:
        st.error("Dataset is empty. Please upload a valid CSV.")
        return

    if show_home:
        render_kpis(df)
        render_data_snapshot(df)
        filtered_df = apply_global_filters(df)
        if filtered_df.empty:
            st.error("Filters removed all rows. Clear some selections to continue.")
            return
        st.session_state["filtered_df"] = filtered_df
        render_home_stage(filtered_df)
        st.info("Grafikleri görmek için soldan bir sunucu seç.")
        return

    filtered_df = st.session_state.get("filtered_df", df)
    if filtered_df.empty:
        st.error("No filtered data available. Return to Ana Sahne to configure filters.")
        return

    render_chart_switcher(presenter_key, filtered_df)


if __name__ == "__main__":
    main()
