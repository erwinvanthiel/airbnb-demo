import torch
from torch import nn
from typing import List
import pandas as pd
import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer
from abc import abstractmethod

# This is a simple fully connected layered NN a.k.a MultiLayer Perceptron
class SimpleNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(SimpleNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return float(x.item())
    
# This wraps the SimpleNNRegressor such that it creates embeddings of the text columns and feeds them to the model
class PreprocessWrapper(mlflow.pyfunc.PythonModel, ABC):
    def __init__(self, model, text_columns: List[str], ohe_columns: List[str], mhe_columns: List[str], embedding_model: SentenceTransformer):
        """
        Args:
        - model: Trained PyTorch, sklearn etc... model 
        - embedding_model: Sentence Transformer model (e.g., "all-MiniLM-L6-v2")
        - text_columns: List of text columns to be transformed into embeddings.
        - ohe_columns: List of columns to be transformed into one-hot-encodings.
        - mhe_columns: List of columns to be transformed into multi-hot-encodings.

        """
        super().__init__()
        self.text_columns = text_columns
        self.ohe_columns = ohe_columns
        self.mhe_columns = mhe_columns
        self.embedding_model = embedding_model
        self.embedding_model.get_sentence_embedding_dimension()
        self.model = model
        self.model.eval()

    def predict(self, model_input: pd.DataFrame) -> float:
        for col in self.text_columns:
            model_input = embed(model_input, self.text_columns, self.embedding_model)
        for col in self.ohe_columns:
            model_input = one_hot_encode(model_input, self.ohe_columns)
        for col in self.mhe_columns:
            model_input = multi_hot_encode(model_input, self.mhe_columns)
        return self.infere(model_input)

    @abstractmethod
    def infere(self, model_input: pd.DataFrame) -> float:
        pass


def embed(input_df: pd.DataFrame, text_columns: List[str], embedding_model: SentenceTransformer) -> pd.DataFrame:
    """Transforms input by replacing text columns with embeddings"""
    for col in text_columns:
        if col not in input_df.columns:
            raise ValueError(f"Column '{col}' not found in input data")

    # Compute text embeddings for each column
    embeddings_list = []
    for col in text_columns:
        text_data = input_df[col].tolist()
        embeddings = embedding_model.encode(text_data, convert_to_numpy=True)
        embeddings_list.append(pd.DataFrame(embeddings))

    # Remove text columns and concatenate embeddings
    df_transformed = input_df.drop(columns=text_columns)
    return pd.concat([df_transformed] + embeddings_list, axis=1)

def one_hot_encode(column_name: string, df: pd.DataFrame) -> pd.DataFrame:
    """One hot encodes a categorical column"""
    categories = np.array(df[column_name])
    encoder = OneHotEncoder()
    ohe = encoder.fit_transform(categories.reshape(-1, 1))     
    df_transformed = input_df.drop(columns=[column_name])  
    return pd.concat([df_transformed] + ohe.toarray(), axis=1)

def multi_hot_encode(column_name: string, df: pd.DataFrame) -> pd.DataFrame:
    """Multi hot encodes a categorical column"""
    unique_values = np.unique(sum(df[column_name], []))
    mhe = np.zeros((len(df), len(unique_values)))
    indices = [np.nonzero(np.isin(unique_values, amenity_array))[0] for amenity_array in df[column_name]]
    for i, indices in enumerate(indices):
        mhe[i, indices] = 1
    df_transformed = df.drop(columns=[column_name])  
    return pd.concat([df_transformed] + mhe, axis=1)

