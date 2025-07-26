#!/bin/bash

# Download MusiCNN model and metadata files
# Based on the Essentia tutorial: https://essentia.upf.edu/tutorial_tensorflow_auto-tagging_classification_embeddings.html

MODEL_DIR="/root/music/playlista/models/musicnn"

# Create directory if it doesn't exist
mkdir -p "$MODEL_DIR"

echo "Downloading MusiCNN model files..."

# Download the model file (.pb)
wget -O "$MODEL_DIR/msd-musicnn-1.pb" \
  "https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.pb"

# Download the metadata file (.json)
wget -O "$MODEL_DIR/msd-musicnn-1.json" \
  "https://essentia.upf.edu/models/autotagging/msd/msd-musicnn-1.json"

echo "Download complete!"
echo "Model files saved to: $MODEL_DIR"
echo "- msd-musicnn-1.pb"
echo "- msd-musicnn-1.json"

# Verify files exist
if [ -f "$MODEL_DIR/msd-musicnn-1.pb" ] && [ -f "$MODEL_DIR/msd-musicnn-1.json" ]; then
    echo "✅ All files downloaded successfully!"
else
    echo "❌ Some files failed to download. Please check manually."
fi 