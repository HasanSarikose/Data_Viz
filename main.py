import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_default_data,
    load_uploaded_data,
    clean_data_drop_incomplete_rows,
)

# SENİN CHART'LARIN
from charts.hasan.overview import render_overview
from charts.hasan.treemap import render_treemap
from charts.hasan.parallel_coordinates import render_parallel_coordinates
from charts.hasan.sankey import render_sankey


st.set_page_config(
    page_title="Shopping Behaviour Dashboard",
    layout="wide"
)


def main():
    st.title("🛒 Shopping Behaviour & Product Ranking Dashboard")

    st.markdown(
        """
        Bu dashboard, müşteri alışveriş davranışlarını incelemek için geliştirildi.  
        Aşağıdan veri yükleyip temizledikten sonra, soldaki hızlı erişim menüsünden
        istediğin grafiğe geçebilirsin.
        """
    )

    st.markdown("### 1️⃣ Data Yükleme")

    # Kullanıcıdan dosya yükleme
    uploaded_file = st.file_uploader(
        "CSV dosyanı yükle (ya da aşağıdan varsayılan dataset'i kullan)",
        type=["csv"]
    )

    use_default = st.checkbox("Varsayılan dataset'i kullan (Kaggle shopping_behavior.csv)")

    df = None
    data_source = None

    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
        data_source = "uploaded"
    elif use_default:
        df = load_default_data()
        data_source = "default"

    if df is None:
        st.info("Lütfen bir CSV dosyası yükle veya varsayılan dataset'i seç.")
        return

    # ==========================
    # 2) DATA TEMİZLEME
    # ==========================
    st.markdown("### 2️⃣ Data Temizleme (Eksik Satırları Sil)")

    if st.checkbox("Eksik verisi olan satırları temizle (drop rows with missing values)", value=True):
        df_clean, before, after = clean_data_drop_incomplete_rows(df)

        st.write(f"Toplam satır (önce): **{before}**")
        st.write(f"Temizlendikten sonra kalan satır: **{after}**")
        st.write(f"Silinen satır sayısı: **{before - after}**")

        df = df_clean
    else:
        st.warning("Dikkat: Eksik değerler temizlenmedi, grafiklerde sorun yaratabilir.")

    st.markdown("---")

    # ==========================
    # 3) HIZLI ERİŞİM / CHART SEÇİCİ
    # ==========================

    st.sidebar.header("📌 Chart Hızlı Erişim")


    chart_registry = {
        "overview": ("Dataset Overview", render_overview),
        "treemap": ("Treemap - Spending by Category", render_treemap),
        "parallel": ("Parallel Coordinates", render_parallel_coordinates),
        "sankey": ("Sankey Diagram", render_sankey)
    }

    chart_keys = list(chart_registry.keys())
    chart_labels = [chart_registry[k][0] for k in chart_keys]

    selected_label = st.sidebar.radio(
        "Görselleştirme Seç",
        chart_labels,
        index=0
    )

    # Seçilen label'a göre ilgili key'i bul
    selected_key = chart_keys[chart_labels.index(selected_label)]
    _, chart_func = chart_registry[selected_key]

    # Seçilen chart'ı render et
    chart_func(df)


if __name__ == "__main__":
    main()
