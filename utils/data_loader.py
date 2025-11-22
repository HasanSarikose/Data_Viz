import pandas as pd
import numpy as np
from typing import Tuple

DEFAULT_DATA_PATH = "data/shopping_behavior.csv"


def load_default_data() -> pd.DataFrame:
    """Varsayılan dataset'i data/ klasöründen yükler."""
    df = pd.read_csv(DEFAULT_DATA_PATH)
    return df


def load_uploaded_data(file) -> pd.DataFrame:
    """Kullanıcının Streamlit üzerinden yüklediği CSV dosyasını okur."""
    df = pd.read_csv(file)
    return df


def clean_data_drop_incomplete_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Her satırda tüm sütunlar dolu mu kontrol eder.
    Eğer herhangi bir sütun eksikse o satırı siler.
    Önce/sonra satır sayılarını da döner.
    """
    df = df.copy()

    before = len(df)
    # En az bir NaN olan satırları sil
    df_clean = df.dropna(how="any")
    after = len(df_clean)

    return df_clean, before, after
