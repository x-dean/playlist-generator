#!/usr/bin/env python3
"""
Main entry point for the playlist generator simple.
This allows running the application with: python -m src
"""

# Suppress TensorFlow warnings globally
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow warnings

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