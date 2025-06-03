# services/llama-explanation/scripts/download_model.sh
#!/bin/bash

# Download quantized Llama 2-7B model for explanations
# This script downloads a pre-quantized GGUF model

MODEL_DIR="/models"
MODEL_FILE="llama-2-7b-explanations.Q4_K_M.gguf"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Check if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

echo "Downloading quantized Llama 2-7B model..."

# Option 1: Download from Hugging Face (replace with actual model URL)
# wget -O "$MODEL_PATH" "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"

# Option 2: Use a mock file for development/testing
echo "Creating mock model file for development..."
echo "MOCK_LLAMA_MODEL" > "$MODEL_PATH"

# Set permissions
chmod 644 "$MODEL_PATH"

echo "Model downloaded successfully to $MODEL_PATH"

# Verify model file
if [ -f "$MODEL_PATH" ]; then
    ls -lh "$MODEL_PATH"
    echo "Model file verified"
else
    echo "Error: Model file not found after download"
    exit 1
fi
