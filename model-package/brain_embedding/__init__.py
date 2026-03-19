"""
brain-embedding — ONNX model files for brain plugin embedder.

This package bundles Snowflake/snowflake-arctic-embed-m-v1.5 ONNX model
so the brain plugin can load embeddings without network access.

Usage:
    import brain_embedding
    model_dir = brain_embedding.get_model_path()
"""

import os

# Model metadata
MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v1.5"
EMBEDDING_DIM = 768
POOLING = "cls"
MODEL_FILE = "onnx/model.onnx"


def get_model_path() -> str:
    """Return the directory containing the ONNX model files."""
    return os.path.join(os.path.dirname(__file__), "model")


def get_model_file() -> str:
    """Return the full path to the ONNX model file."""
    return os.path.join(get_model_path(), MODEL_FILE)


def is_available() -> bool:
    """Check if the model files are present."""
    return os.path.isfile(get_model_file())
