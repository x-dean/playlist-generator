#!/usr/bin/env python3
"""
Test script to check MusicNN model paths and availability.
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.core.config_loader import ConfigLoader
from core.logging_setup import log_universal

def test_musicnn_paths():
    """Test MusiCNN model file paths and accessibility."""
    
    print("=== MusiCNN Path Verification ===")
    
    # Load configuration
    config = ConfigLoader().get_audio_analysis_config()
    
    # Get paths from config
    model_path = config.get('MUSICNN_MODEL_PATH', '/app/models/msd-musicnn-1.pb')
    json_path = config.get('MUSICNN_JSON_PATH', '/app/models/msd-musicnn-1.json')
    
    print(f"Configuration paths:")
    print(f"  Model path: {model_path}")
    print(f"  JSON path: {json_path}")
    
    # Check if paths exist
    print(f"\nFile existence check:")
    print(f"  Model file exists: {os.path.exists(model_path)}")
    print(f"  JSON file exists: {os.path.exists(json_path)}")
    
    # Check if directories exist
    model_dir = os.path.dirname(model_path)
    json_dir = os.path.dirname(json_path)
    
    print(f"\nDirectory check:")
    print(f"  Model directory exists: {os.path.exists(model_dir)}")
    print(f"  JSON directory exists: {os.path.exists(json_dir)}")
    
    # List contents of model directory if it exists
    if os.path.exists(model_dir):
        print(f"\nContents of {model_dir}:")
        try:
            files = os.listdir(model_dir)
            for file in files:
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size} bytes)")
                else:
                    print(f"  {file} (directory)")
        except Exception as e:
            print(f"  Error listing directory: {e}")
    
    # Check file permissions
    print(f"\nFile permissions:")
    for path, name in [(model_path, "Model"), (json_path, "JSON")]:
        if os.path.exists(path):
            try:
                # Check if readable
                with open(path, 'r') as f:
                    f.read(1)
                print(f"  {name} file is readable")
            except Exception as e:
                print(f"  {name} file is NOT readable: {e}")
        else:
            print(f"  {name} file does not exist")
    
    # Test TensorFlow availability
    print(f"\nTensorFlow check:")
    try:
        import tensorflow as tf
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        print(f"  TensorFlow version: {tf.__version__}")
        print(f"  TensorFlow available: True")
    except ImportError:
        print(f"  TensorFlow available: False")
    
    # Test model loading
    print(f"\nModel loading test:")
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            # Suppress TensorFlow warnings during model loading
            tf.get_logger().setLevel('ERROR')
            tf.autograph.set_verbosity(0)
            if model_path.endswith('.pb'):
                model = tf.saved_model.load(model_path)
                print(f"  ✓ Model loaded successfully (SavedModel)")
            elif model_path.endswith('.h5'):
                model = tf.keras.models.load_model(model_path)
                print(f"  ✓ Model loaded successfully (Keras)")
            else:
                print(f"  ❌ Unknown model format: {model_path}")
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
    else:
        print(f"  ❌ Model file not found: {model_path}")
    
    # Test JSON loading
    print(f"\nJSON configuration test:")
    if os.path.exists(json_path):
        try:
            import json
            with open(json_path, 'r') as f:
                config_data = json.load(f)
            print(f"  ✓ JSON loaded successfully")
            if 'tag_names' in config_data:
                print(f"  ✓ Tag names found: {len(config_data['tag_names'])} tags")
            else:
                print(f"  ⚠️  No tag_names found in JSON")
        except Exception as e:
            print(f"  ❌ Failed to load JSON: {e}")
    else:
        print(f"  ❌ JSON file not found: {json_path}")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_musicnn_paths() 