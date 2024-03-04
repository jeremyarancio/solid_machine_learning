import pandas as pd


PATH = "data/data.parquet"


if __name__ == "__main__":

    df = pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": ["a", "a", "b", "b", "c"],
        "feature_c": [0, 0, None, 1, 1]
    })

    df.to_parquet(PATH)