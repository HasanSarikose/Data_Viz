# charts/fng/sankey.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List


def build_sankey_from_columns(df: pd.DataFrame, cols: List[str]):
    """
    Seçilen kolonlar arasındaki akışı (flow) hesaplar.
    cols: örn. ["customer_segment", "category", "gender"]
    """
    # Tüm node label'larını topla
    labels = []
    for col in cols:
        labels.extend(df[col].astype(str).unique().tolist())
    labels = list(dict.fromkeys(labels))  # unique + order preserved

    label_to_idx = {label: i for i, label in enumerate(labels)}

    sources = []
    targets = []
    values = []

    # Önce tüm kombinasyonların count'unu al
    grouped = df[cols].astype(str).groupby(cols).size().reset_index(name="count")

    # Her komşu kolon çifti için link oluştur
    for i in range(len(cols) - 1):
        c1 = cols[i]
        c2 = cols[i + 1]

        pair_grouped = grouped.groupby([c1, c2])["count"].sum().reset_index()

        for _, row in pair_grouped.iterrows():
            src_label = row[c1]
            tgt_label = row[c2]
            cnt = int(row["count"])

            sources.append(label_to_idx[src_label])
            targets.append(label_to_idx[tgt_label])
            values.append(cnt)

    return labels, sources, targets, values


def render_sankey(df: pd.DataFrame):
    st.header("🔗 Sankey Diagram (Dynamic Categorical Flow)")

    if df.empty:
        st.error("Geçerli bir veri bulunamadı.")
        return

    # Kategorik kolonları seç
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(cat_cols) < 2:
        st.error("Sankey için en az 2 kategorik kolon gerekli.")
        return

    st.markdown("Akış için kullanmak istediğin kolonları seç:")

    selected_cols = st.multiselect(
        "🔢 Categorical columns (2–3 önerilir)",
        cat_cols,
        default=cat_cols[:3] if len(cat_cols) >= 3 else cat_cols[:2],
    )

    if len(selected_cols) < 2:
        st.warning("En az iki kolon seçmelisin.")
        return

    if len(selected_cols) > 4:
        st.info("⚠ Çok fazla kolon seçmek Sankey'i karışık hale getirebilir (2–3 ideal).")

    # Veriyi biraz kısıtlayarak aşırı karmaşayı azalt (opsiyonel)
    max_rows = 5000
    if len(df) > max_rows:
        df_use = df.sample(max_rows, random_state=42)
        st.caption(f"Veri çok büyük olduğu için rastgele {max_rows} satır üzerinde işlem yapılıyor.")
    else:
        df_use = df

    try:
        labels, sources, targets, values = build_sankey_from_columns(df_use, selected_cols)
    except Exception as e:
        st.error(f"Linkler oluşturulurken hata oluştu: {e}")
        return

    if len(sources) == 0:
        st.error("Seçilen kolon kombinasyonları için akış bulunamadı.")
        return

    link = dict(
        source=sources,
        target=targets,
        value=values,
    )

    node = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels
    )

    sankey = go.Sankey(node=node, link=link)

    fig = go.Figure(data=[sankey])
    fig.update_layout(
        title_text=f"Sankey Diagram – {' → '.join(selected_cols)}",
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)
