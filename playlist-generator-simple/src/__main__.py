#!/usr/bin/env python3
"""
Main entry point for the playlist generator simple.
This allows running the application with: python -m src
"""

# Suppress TensorFlow warnings globally
import os
# Configure TensorFlow logging BEFORE any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING, show only ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU to avoid GPU-related warnings

# Suppress TensorFlow logging after import
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
except ImportError:
    pass

from .enhanced_cli import main

if __name__ == "__main__":
    exit(main()) 