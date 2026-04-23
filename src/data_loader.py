import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():

    df = pd.read_csv("data/raw/bug_dataset_50k.csv")  # change filename if needed

    # keep only required columns
    df = df[['description', 'severity']].dropna()

    # rename columns to match pipeline
    df.columns = ['Description', 'Label']

    # split dataset
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    return train, test