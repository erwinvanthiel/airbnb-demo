import torch
from torch import nn
from typing import List
import pandas as pd
import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer

# This is a simple fully connected layered NN a.k.a MultiLayer Perceptron
class SimpleNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(SimpleNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# This wraps the SimpleNNRegressor such that it creates embeddings of the text columns and feeds them to the model  
class SimpleNNRegressorEmbeddingWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model: SimpleNNRegressor, text_columns: List[str], embedding_model: SentenceTransformer):
        """
        Args:
        - model: Trained PyTorch model (SimpleNNRegressor)
        - embedding_model: Sentence Transformer model (e.g., "all-MiniLM-L6-v2")
        - text_columns: List of text columns to be transformed into embeddings.
        """
        super().__init__()
        self.text_columns = text_columns
        self.embedding_model = embedding_model
        self.embedding_model.get_sentence_embedding_dimension()
        self.model = model
        self.model.eval()

    def predict(self, model_input: pd.DataFrame):

        # Convert to tensor
        input_tensor = torch.tensor(embed(model_input).values, dtype=torch.float32)

        # Run model inference
        with torch.no_grad():
            predictions = self.model(input_tensor).numpy().flatten()

        return predictions

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
