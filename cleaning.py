import numpy as np
import pandas as pd

def ordEnc(df, columns_to_encode):
    # make a copy of the original df to avoid changing the original data and also avoid SettingWithCopyWarning
    df_enc = df.copy()
    mappings = {}
    for col in columns_to_encode:

        categories = df_enc[col].dropna().unique()
        sorted_cats = sorted(categories)

        mapping = {cat : i for i, cat in enumerate(sorted_cats)}
        mappings[col] = mapping

        new_col_name = f"{col}_encoded"
        df_enc[new_col_name] = df_enc[col].map(mapping)

    return df_enc, mappings

def oneHotEnc(df, columns_to_encode):
    df_enc = df.copy()  
    for col in columns_to_encode:
        categories = df_enc[col].dropna().unique()
        for category in categories:
            df_enc[f"{col}_{category}"] = (df_enc[col] == category).astype(int)
        
    df_enc = df_enc.drop(columns = columns_to_encode)

    return df_enc


# median and mode can also be used
def impute(df, columns_to_impute):
    df_imp = df.copy()
    for col in columns_to_impute:
        # if all values in a column are NAN
        if df_imp[col].isna.sum() == len(df_imp[col]):
            continue
        # to implement .mean() only on numerical columns
        if pd.api.types.is_numeric_dtype(df_imp[col]):
            df_imp[col] = df_imp[col].fillna(df_imp[col].mean())
        # for categorical columns -> replace with most frequent value using .mode()
        else:
            df_imp[col] = df_imp[col].fillna(df_imp[col].mode()[0])
        
    return df_imp