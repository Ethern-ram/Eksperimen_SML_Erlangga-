import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import argparse

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    print("Preprocessing data...")
    df = df.drop('id', axis=1)
    df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
    df = df.drop('diagnosis', axis=1)

    X = df.drop('target', axis=1)
    y = df['target']
    
    # Validation
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    df_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    df_train['target'] = y_train.values
    
    df_test = pd.DataFrame(X_test_scaled, columns=X.columns)
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
    parser.add_argument("--input", type=str, default="../breast_cancer_raw/breast_cancer.csv", help="Input dataset path")
    parser.add_argument("--output", type=str, default="../breast_cancer_preprocessing", help="Output directory path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    df = load_data(args.input)
    df_train, df_test = preprocess_data(df)
    save_data(df_train, df_test, args.output)
    print("Automated preprocessing completed.")
