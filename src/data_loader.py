import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():

    df = pd.read_csv("data/raw/bug_dataset_50k.csv")

    df = df[
        [
            "title",
            "description",
            "bug_category",
            "bug_domain",
            "tech_stack",
            "environment",
            "developer_role",
            "severity",
        ]
    ].dropna()

    df["text"] = (
        df["title"] + " "
        + df["description"] + " "
        + df["bug_category"] + " "
        + df["bug_domain"] + " "
        + df["tech_stack"] + " "
        + df["environment"] + " "
        + df["developer_role"]
    )

    df = df[["text", "severity"]]

    df.columns = ["Description", "Label"]

    df["Label"] = df["Label"].str.lower()

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Label"],
    )

    return train, test