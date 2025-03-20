from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator
from typing import List, Tuple
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def embed(df: pd.DataFrame, text_columns: List[str], embedding_model: SentenceTransformer) -> pd.DataFrame:
    """Transforms input by replacing text columns with embeddings"""
    for col in text_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in input data")

    # Compute text embeddings for each column
    embeddings_list = []
    for col in text_columns:
        text_data = df[col].tolist()
        embeddings = embedding_model.encode(text_data, convert_to_numpy=True)
        embeddings_list.append(pd.DataFrame(embeddings))

    # Remove text columns and concatenate embeddings
    df_transformed = df.drop(columns=text_columns)
    return pd.concat([df_transformed] + embeddings_list, axis=1)

def one_hot_encode(column_name: str, df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """One-hot encodes a categorical column, assigning class names as column names."""
    categories = df[[column_name]].to_numpy()
    ohe = encoder.transform(categories)
    df_transformed = df.drop(columns=[column_name]).reset_index(drop=True)
    ohe_df = pd.DataFrame(ohe.toarray(), columns=encoder.get_feature_names_out([column_name])).reset_index(drop=True)
    return pd.concat([df_transformed, ohe_df], axis=1)

def multi_hot_encode(column_name: str, df: pd.DataFrame, encoder: MultiLabelBinarizer) -> pd.DataFrame:
    """Multi hot encodes a categorical column"""
    multi_hot_encoded = encoder.transform(np.array(df[[column_name]]))
    mhe_df = pd.DataFrame(multi_hot_encoded, columns=encoder.classes_)
    df_transformed = df.drop(columns=[column_name])
    return pd.concat([df_transformed, mhe_df], axis=1)