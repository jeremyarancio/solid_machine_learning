import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO)


# Bad code
def process(path: str, output_path: str) -> pd.DataFrame:
    """"""
    df = pd.read_parquet(path)
    logging.info(f"Data: {df}")
    
    # Normalization
    std = np.std(df["feature_a"])
    mean = np.mean(df["feature_a"])
    normalized_feature = (df["feature_a"] - mean) / std

    # Categorical value
    encoder = LabelEncoder()
    encoded_feature = encoder.fit_transform(df["feature_b"])
    
    # Nan
    filled_feature = df["feature_c"].fillna(-1)

    processed_df = pd.DataFrame({
        "feature_a": normalized_feature,
        "feature_b": encoded_feature,
        "feature_c": filled_feature
    }) 
    logging.info(f"Processed data: {processed_df}")

    processed_df.to_parquet(output_path)


def main():
    path = "data/data.parquet"
    output_path = "data/preprocessed_data.parquet"
    process(path, output_path)


if __name__ == "__main__":
    main()