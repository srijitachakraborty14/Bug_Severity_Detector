import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():

    # Dynamic path handling
    file_path = os.path.abspath("data/raw/bugs.csv")
    print("Reading dataset from:", file_path)

    df = pd.read_csv(file_path, encoding="latin-1")

    # Keep and clean core target features
    df = df[["Description", "Severity", "Priority"]].dropna()

    df["Severity"] = df["Severity"].str.lower()
    df["Priority"] = df["Priority"].str.lower()

    # Feature Engineering
    df["text"] = df["Description"] + " " + df["Priority"]

    # Restructure to match pipeline expectations
    df = df[["text", "Severity"]]
    df.columns = ["Description", "Label"]

    # ==========================
    # REMOVE NORMAL CLASS
    # ==========================
    df = df[df["Label"] != "normal"]
    major_df = df[df["Label"] == "major"].sample(
    n=400,
    random_state=42
    )

    other_df = df[df["Label"] != "major"]

    df = pd.concat([major_df, other_df])

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\n===== LABEL DISTRIBUTION =====")
    print(df["Label"].value_counts())

    # Train-Test Split
    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"]
    )

    return train, test