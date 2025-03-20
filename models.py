import torch
from torch import nn
from typing import List, Tuple
import pandas as pd
import numpy as np
import mlflow
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator
import pickle
import joblib
import os
from pathlib import Path
from utils import embed, one_hot_encode, multi_hot_encode

# This is a simple fully connected layered NN a.k.a MultiLayer Perceptron
class SimpleNNRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(SimpleNNRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
# This is a MLFlow wrapper for a model. It stores and loads the required models and performs the required preprocessing steps. Loading of the predictive model and the inference method is to be implemented by concrete child because sklearn and pytorch models for exmaple have different inference and loading mechanisms. It works with an artifacts object that depicts the locations of the stored models. This artifact object is used to load the model dependencies when its served by MLFlow.
class PreprocessWrapper(mlflow.pyfunc.PythonModel, ABC):
    def __init__(self, text_columns: List[str], ohe_columns: List[Tuple[str, OneHotEncoder]], mhe_columns: List[Tuple[str, MultiLabelBinarizer]], embedding_model_path: str, scale_columns: List[Tuple[str, MinMaxScaler]], artifact_base_path: str):
        """
        Initializes the wrapper and stores preprocessing artifacts.

        Args:
            text_columns (List[str]): List of text columns requiring embedding.
            ohe_columns (List[Tuple[str, OneHotEncoder]]): One-Hot Encoders and corresponding column names.
            mhe_columns (List[Tuple[str, MultiLabelBinarizer]]): Multi-Label Binarizers and corresponding column names.
            embedding_model_path (str): Path to embedding model (if applicable).
            scale_columns (List[Tuple[str, MinMaxScaler]]): MinMax Scalers and corresponding column names.
            artifact_base_path (str): Base path for storing artifacts.
        """
        super().__init__()
        self.text_columns = text_columns
        self.ohe_columns = ohe_columns
        self.mhe_columns = mhe_columns
        self.embedding_model_path = embedding_model_path
        self.model = None
        self.embedding_model = None

        # the file paths for the sklearn models
        self.scaler_paths: List[Tuple[str, str]] = []
        self.onehot_encoder_paths: List[Tuple[str, str]] = []
        self.multihot_encoder_paths: List[Tuple[str, str]] = []

        # the loaded models
        self.scalers: List[Tuple[str, BaseEstimator]] = scale_columns
        self.onehot_encoders: List[Tuple[str, OneHotEncoder]] = ohe_columns
        self.multihot_encoders: List[Tuple[str, MultiLabelBinarizer]] = mhe_columns
        self.artifact_base_path = artifact_base_path

        # store the scalers
        for col_name, scaler in scale_columns:
            scaler_path = (Path(artifact_base_path) / "models" /  f"{col_name}_scaler.pkl").resolve()
            os.makedirs(scaler_path.parent, exist_ok=True)
            scaler_path = str(scaler_path)
            joblib.dump(scaler, scaler_path)
            self.scaler_paths.append((col_name, scaler_path))

        # store the ohe-encoders
        for col_name, ohe_encoder in ohe_columns:
            onehot_encoder_path = (Path(artifact_base_path) / "models" /  f"{col_name}_ohe_encoder.pkl").resolve()
            os.makedirs(onehot_encoder_path.parent, exist_ok=True)
            onehot_encoder_path = str(onehot_encoder_path)
            joblib.dump(ohe_encoder, onehot_encoder_path)
            self.onehot_encoder_paths.append((col_name, onehot_encoder_path))

        # store the mhe-encoders
        for col_name, mhe_encoder in mhe_columns:
            multihot_encoder_path = (Path(artifact_base_path) / "models" / f"{col_name}_mhe_encoder.pkl").resolve()
            os.makedirs(multihot_encoder_path.parent, exist_ok=True)
            multihot_encoder_path = str(multihot_encoder_path)
            joblib.dump(mhe_encoder, multihot_encoder_path)
            self.multihot_encoder_paths.append((col_name, multihot_encoder_path))

    def load_context(self, context):
        """
        Loads preprocessing models and metadata from MLflow artifacts.

        Args:
            context: MLflow context containing artifact paths.
        """
        metadata_path = context.artifacts["metadata"]
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get paths from metadata
        self.ohe_columns = metadata["ohe_columns"]
        self.mhe_columns = metadata["mhe_columns"]
        self.scaler_paths = metadata["scaler_paths"]

        # Load models from path
        self.scalers = [(col, joblib.load(path)) for col, path in self.scaler_paths]
        self.onehot_encoders = [(col, joblib.load(path)) for col, path in self.onehot_encoder_paths]
        self.multihot_encoders = [(col, joblib.load(path)) for col, path in self.multihot_encoder_paths]
        self.load_model_from_path(metadata["model_path"])        
        
        # Load embedding model if present
        if "embedding_model_path" in context.artifacts:
            self.embedding_model = SentenceTransformer(context.artifacts["embedding_model"])
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.Series:
        """
        Applies preprocessing steps and makes predictions.

        Args:
            context: MLflow context.
            model_input (pd.DataFrame): Input data.

        Returns:
            pd.Series: Model predictions.
        """
        processed_model_input = model_input
        for (col, scaler) in self.scalers:
            processed_model_input[col] = scaler.transform(processed_model_input[[col]])
        for col in self.text_columns:
            if self.embedding_model_path:
                processed_model_input = embed(processed_model_input, self.text_columns, self.embedding_model)
            processed_model_input = processed_model_input.drop(columns=[col])
        for col, encoder in self.onehot_encoders:
            processed_model_input = one_hot_encode(col, processed_model_input, encoder)
        for col, encoder in self.multihot_encoders:
            processed_model_input = multi_hot_encode(col, processed_model_input, encoder)
        return self.infere(processed_model_input)
    
    @abstractmethod
    def infere(self, model_input: pd.DataFrame) -> pd.Series:
        """Perform inference using the loaded model."""
        pass

    @abstractmethod
    def load_model_from_path(self, model_path: str):
        """Load the model from the given path."""
        pass

# This is an mflow model wrapper for an sklearn regression model
class SklearnRegressorPreprocessWrapper(PreprocessWrapper):
    def __init__(self, model: BaseEstimator, text_columns: List[str], ohe_columns: List[Tuple[str, OneHotEncoder]], mhe_columns: List[Tuple[str, MultiLabelBinarizer]], scale_columns: List[Tuple[str, MinMaxScaler]], artifact_base_path: str):
        """Save model to disk and initialize wrapper with model path."""
        self.model_path = (Path(artifact_base_path) / "models" /  f"{type(model).__name__}.pkl").resolve()
        os.makedirs(self.model_path.parent, exist_ok=True)
        self.model_path = str(self.model_path)
        joblib.dump(model, self.model_path)
        super().__init__(text_columns, ohe_columns, mhe_columns, None, scale_columns, artifact_base_path)
        self.model = model

    def infere(self, model_input: pd.DataFrame) -> pd.Series:
        """Perform inference using the loaded scikit-learn regression model."""
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        predictions = self.model.predict(model_input)
        return pd.Series(predictions, index=model_input.index)

    def load_model_from_path(self, path):
        self.model: BaseEstimator = joblib.load(model_path)

# This is an mflow model wrapper for a pytorch DNN regression model
class TorchNNPreprocessWrapper(PreprocessWrapper):
    def __init__(self, model: torch.nn.Module, text_columns: List[str], ohe_columns: List[Tuple[str, OneHotEncoder]], mhe_columns: List[Tuple[str, MultiLabelBinarizer]], embedding_model_path: str, scale_columns: List[Tuple[str, MinMaxScaler]], artifact_base_path: str):
        """Save model to disk and initialize wrapper with model path."""
        self.model_path = (Path(artifact_base_path) / "models" /  f"{type(model).__name__}.pkl").resolve()
        os.makedirs(self.model_path.parent, exist_ok=True)
        self.model_path = str(self.model_path)
        joblib.dump(model, self.model_path)
        super().__init__(text_columns, ohe_columns, mhe_columns, embedding_model_path, scale_columns, artifact_base_path)
        self.model = model

    def infere(self, model_input: pd.DataFrame) -> pd.Series:
        """Perform inference using the loaded PyTorch model."""
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        # Convert DataFrame to Tensor
        input_tensor = torch.tensor(model_input.values, dtype=torch.float32)
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        with torch.no_grad():  # Disable gradient computation for inference
            predictions = self.model(input_tensor)
        
        # Convert predictions to pandas Series
        return pd.Series(predictions.numpy().flatten(), index=model_input.index)

    def load_model_from_path(self, path: str):
        """Load a PyTorch model from a file."""
        self.model = torch.load(path)  # Load the saved PyTorch model