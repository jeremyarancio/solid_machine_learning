from solid.open_close import Processor, NaFiller
from solid.single_responsability import load_data, save_df, compose_df


def process(
        path: str,
        output_path: str,
        numerical_processor: Processor,
        categorical_processor: Processor,
        na_filler: NaFiller
) -> None:
    """"""
    df = load_data(path=path)
    normalized_feature = numerical_processor.process(df["feature_a"])
    encoded_feature = categorical_processor.process(df["feature_b"])
    filled_feature = na_filler.process(df["feature_c"])
    processed_df = compose_df(
        normalized_feature, 
        encoded_feature, 
        filled_feature,
        column_names=df.columns
    )
    save_df(df=processed_df, path=output_path)