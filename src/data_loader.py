import pandas as pd
from sklearn.model_selection import train_test_split
import os


def load_data():

    file_path = os.path.abspath("data/raw/bugs.csv") #file path

    print("Reading dataset from:", file_path)

    df = pd.read_csv(file_path, encoding="latin-1")

    df = df[["Description", "Severity", "Priority"]].dropna()

    df["Severity"] = df["Severity"].str.lower()
    df["Priority"] = df["Priority"].str.lower()

    df["text"] = df["Description"] + " " + df["Priority"]

    df = df[["text", "Severity"]]
    df.columns = ["Description", "Label"]

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"]
    )

    return train, test