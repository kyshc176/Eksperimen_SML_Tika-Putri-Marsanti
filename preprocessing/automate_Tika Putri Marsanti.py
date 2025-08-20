import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_and_preprocess(df):
    # Bersihkan nama kolom dari spasi
    df.columns = df.columns.str.strip()

    # Hapus baris yang memiliki nilai kosong
    df = df.dropna().copy()

    # Normalisasi kolom GENDER
    df['GENDER'] = df['GENDER'].replace({
        'M': 'L',
        'F': 'P',
        'MALE': 'L',
        'FEMALE': 'P'
    })

    # Label encoding untuk kolom GENDER
    le_gender = LabelEncoder()
    df['GENDER'] = le_gender.fit_transform(df['GENDER'])

    # Label encoding untuk kolom target: LUNG_CANCER
    df['LUNG_CANCER'] = df['LUNG_CANCER'].replace({'YES': 1, 'NO': 0})

    # Pastikan semua kolom selain AGE bertipe integer
    for col in df.columns:
        if col != 'AGE':
            df[col] = df[col].astype(int)

    return df

def preprocess(filepath):
    df = load_data(filepath)
    processed_data = clean_and_preprocess(df)
    return processed_data

if __name__ == "__main__":
    raw_path = 'namadataset_raw/survey_lung_cancer_raw.csv'  # path dataset mentah
    processed_dir = 'preprocessing/namadataset_preprocessing'
    processed_path = os.path.join(processed_dir, 'survey_lung_cancer_processed.csv')

    # Buat folder penyimpanan jika belum ada
    os.makedirs(processed_dir, exist_ok=True)

    # Jalankan preprocessing dan simpan hasil
    data_ready = preprocess(raw_path)
    data_ready.to_csv(processed_path, index=False)

    print(f"✅ Data berhasil diproses dan disimpan di: {processed_path}")
