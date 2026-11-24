import pandas as pd

from config import DATA_PATH, CLEAN_DATA_PATH


def prepare_data():

    df = pd.read_csv(DATA_PATH)

    print(f"Raw data: {len(df)} rows")

    # Remove missing values
    df = df.dropna(
        subset=[
            "id", "title", "year",
            "genre", "overview"
        ]
    ).copy()

    # Remove extra whitespace
    for col in ["title", "genre", "overview"]:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]

    # Remove duplicate records
    df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(
        subset=["title", "year"]
    )

    # Combine text for embedding
    df["text"] = (
        df["title"] + " | " +
        df["genre"] + " | " +
        df["overview"]
    )

    # Save processed dataset
    df.to_csv(
        CLEAN_DATA_PATH,
        index=False,
        encoding="utf-8"
    )

    print(f"Clean data: {len(df)} rows")
    print("Data preparation completed!")

    return df


if __name__ == "__main__":
    prepare_data()
