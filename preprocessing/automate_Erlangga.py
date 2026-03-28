import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_data():
    """Load raw Titanic dataset dari KaggleHub"""
    print("Loading Titanic data via KaggleHub...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "brendan45774/test-file",
        "tested.csv"
    )
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_data(df):
    """Preprocessing identik dengan notebook Eksperimen_Erlangga.ipynb"""
    print("Preprocessing data...")
    df_clean = df.copy()

    # 1. Hapus kolom yang tidak relevan
    cols_to_drop = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df_clean.columns]
    df_clean = df_clean.drop(columns=cols_to_drop)

    # 2. Handle Missing Values
    # Age: isi dengan median
    if 'Age' in df_clean.columns:
        df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    # Embarked: isi dengan modus
    if 'Embarked' in df_clean.columns:
        df_clean['Embarked'] = df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0])
    # Fare: isi dengan median
    if 'Fare' in df_clean.columns:
        df_clean['Fare'] = df_clean['Fare'].fillna(df_clean['Fare'].median())

    # Drop sisa baris yang masih null
    df_clean = df_clean.dropna()
    df_clean = df_clean.drop_duplicates()

    # 3. Encoding kolom kategorikal
    le = LabelEncoder()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # 4. Pisahkan fitur dan target (kolom target: 'Survived')
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']

    # 5. Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Gabung kembali fitur + target, simpan kolom 'Survived'
    df_train = X_train.copy()
    df_train['Survived'] = y_train.values

    df_test = X_test.copy()
    df_test['Survived'] = y_test.values

    print(f"Train set: {df_train.shape}, Test set: {df_test.shape}")
    return df_train, df_test

def save_data(df_train, df_test, out_dir):
    """Simpan hasil preprocessing ke folder output"""
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    test_path  = os.path.join(out_dir, "test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Data saved to: {out_dir}")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

if __name__ == "__main__":
    # Output relative dari lokasi script ini
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "..", "titanic_preprocessing")
    out_dir = os.path.normpath(out_dir)

    df = load_data()
    df_train, df_test = preprocess_data(df)
    save_data(df_train, df_test, out_dir)
    print("Automated preprocessing completed successfully!")