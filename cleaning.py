import numpy as np
import pandas as pd

def ordEnc(df, columns_to_encode):

    df_enc = df.copy()
    mappings = {}

    for column_name in columns_to_encode:

        categories = df_enc[column_name].dropna().unique()
        sorted_cats = sorted(categories)

        mapping = {cat : i for i, cat in enumerate(sorted_cats)}
        mappings[column_name] = mapping

        new_col_name = f"{column_name}_encoded"
        df_enc[new_col_name] = df_enc[column_name].map(mapping)

    return df_enc, mappings


def oneHotEnc(df, columns_to_encode):

    df_enc = df.copy()
    
    for column_name in columns_to_encode:

        categories = df[column_name].dropna().unique()

        for category in categories:
            df_enc[f"{column_name}_{category}"] = (df_enc[column_name] == category).astype(int)
        
    df_enc = df_enc.drop(columns = columns_to_encode)

    return df_enc


# median and mode can also be used
def impute(df, columns_to_impute):
    
    df_imp = df.copy()

    for column_name in columns_to_impute:

        df_imp[column_name] = df_imp[column_name].fillna(df_imp[column_name].mean())

    return df_imp