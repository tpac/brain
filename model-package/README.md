# brain-arctic-embed

ONNX model files for the [brain plugin](https://github.com/tpac/brain) embedder.

Bundles **Snowflake/snowflake-arctic-embed-m-v1.5** (768d, CLS pooling, Apache-2.0 license) so the brain plugin can load embeddings without network access.

## Usage

```python
import brain_arctic_embed

model_dir = brain_arctic_embed.get_model_path()
# Pass to FastEmbed or ONNX runtime
```

## Building

1. Download model files from HuggingFace:
   ```bash
   bash download-model.sh
   ```
2. Build wheel:
   ```bash
   pip install build
   python -m build
   ```
3. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```
