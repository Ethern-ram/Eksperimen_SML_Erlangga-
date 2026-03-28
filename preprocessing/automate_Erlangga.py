import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import argparse
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_data(filepath=None):
    print("Loading data via KaggleHub...")
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "brendan45774/test-file",
        "tested.csv"
    )
    # Save raw to titanic_raw as well
    os.makedirs("../titanic_raw", exist_ok=True)
    df.to_csv("../titanic_raw/tested.csv", index=False)
    return df

def preprocess_data(df):
    print("Preprocessing data (Titanic)...")
    df_clean = df.copy()
    
    # 1. Handle Missing Values
    df_clean = df_clean.dropna()
    
    # 2. Hapus Data Duplikat
    df_clean = df_clean.drop_duplicates()
    
    # 3. Encoding Data Kategorikal
    le = LabelEncoder()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = le.fit_transform(df_clean[col])
        
    # 4. Split Data
    X = df_clean.drop('Survived', axis=1)
    y = df_clean['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    df_train = X_train.copy()
    df_train['target'] = y_train.values  # Re-assign target column for unification
    
    df_test = X_test.copy()
    df_test['target'] = y_test.values
    
    return df_train, df_test

def save_data(df_train, df_test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    test_path = os.path.join(out_dir, "test.csv")
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print(f"Data saved to {out_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Automate Preprocessing for MSML")
    parser.add_argument("--output", type=str, default="../titanic_preprocessing", help="Output directory path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df = load_data()
    df_train, df_test = preprocess_data(df)
    save_data(df_train, df_test, args.output)
    print("Automated preprocessing completed.")
