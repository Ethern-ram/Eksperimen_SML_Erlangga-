import os
import json
import urllib.request

def init_project():
    base_dir = r"d:\Pijak\PROJECT\SMSML_Erlangga Pradana Kurniawan\Eksperimen_SML_Erlangga"
    raw_dir = os.path.join(base_dir, "breast_cancer_raw")
    prep_dir = os.path.join(base_dir, "preprocessing")
    workflow_dir = os.path.join(base_dir, ".github", "workflows")
    
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(workflow_dir, exist_ok=True)
    
    # 1. Download raw data directly
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    raw_path = os.path.join(raw_dir, "breast_cancer.csv")
    try:
        urllib.request.urlretrieve(url, raw_path)
        print(f"Saved raw data to {raw_path}")
        
        # Adding headers to the downloaded CSV
        with open(raw_path, 'r') as f:
            lines = f.readlines()
        
        # wdbc.data has 32 columns: ID, Diagnosis, 30 features
        headers = ["id", "diagnosis"] + [f"feature_{i}" for i in range(1, 31)]
        with open(raw_path, 'w') as f:
            f.write(",".join(headers) + "\n")
            f.writelines(lines)
            
    except Exception as e:
        print("Error downloading: ", e)
    
    # 2. Create Notebook (Using the WDBC header structure)
    notebook_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Eksperimen Data - Sistem Machine Learning\n", "Tahapan ini melakukan pemuatan data, eksplorasi (EDA), dan preprocessing."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "import os"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Data Loading"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "raw_data_path = '../breast_cancer_raw/breast_cancer.csv'\n",
                    "df = pd.read_csv(raw_data_path)\n",
                    "df = df.drop('id', axis=1) # Drop ID column\n",
                    "# Convert diagnosis to binary target: M=1 (Malignant), B=0 (Benign)\n",
                    "df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})\n",
                    "df = df.drop('diagnosis', axis=1)\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Exploratory Data Analysis (EDA)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(df.info())\n",
                    "print(df.describe())"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sns.countplot(x='target', data=df)\n",
                    "plt.title('Target Distribution')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3. Preprocessing"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "X = df.drop('target', axis=1)\n",
                    "y = df['target']\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "X_train_scaled = scaler.fit_transform(X_train)\n",
                    "X_test_scaled = scaler.transform(X_test)\n",
                    "\n",
                    "df_train = pd.DataFrame(X_train_scaled, columns=X.columns)\n",
                    "df_train['target'] = y_train.values\n",
                    "\n",
                    "df_test = pd.DataFrame(X_test_scaled, columns=X.columns)\n",
                    "df_test['target'] = y_test.values"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "os.makedirs('../breast_cancer_preprocessing', exist_ok=True)\n",
                    "df_train.to_csv('../breast_cancer_preprocessing/train.csv', index=False)\n",
                    "df_test.to_csv('../breast_cancer_preprocessing/test.csv', index=False)\n",
                    "print('Preprocessing selesai.')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.8.5"}
        },
        "nbformat": 4, "nbformat_minor": 4
    }
    nb_path = os.path.join(prep_dir, "Eksperimen_Erlangga.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=4)
    print(f"Created notebook at {nb_path}")

if __name__ == '__main__':
    init_project()
