#!/bin/bash
# Download snowflake-arctic-embed-m-v1.5 ONNX model files from HuggingFace.
# Run this on a machine with internet access (your Mac), then build the wheel.
set -euo pipefail

MODEL_DIR="brain_arctic_embed/model"
HF_REPO="Snowflake/snowflake-arctic-embed-m-v1.5"

echo "Downloading $HF_REPO model files..."

pip3 install huggingface_hub --quiet

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$HF_REPO',
    local_dir='$MODEL_DIR',
    allow_patterns=[
        'onnx/model.onnx',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'config.json',
    ],
)
print('Done.')
"

echo ""
echo "Model files downloaded to $MODEL_DIR"
echo "Size: $(du -sh $MODEL_DIR | cut -f1)"
echo ""
echo "Next steps:"
echo "  pip3 install build twine"
echo "  python3 -m build"
echo "  python3 -m twine upload dist/*"
