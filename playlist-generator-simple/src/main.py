"""
Main entry point for Playlist Generator Simple.
Supports both CLI and API modes.
"""

import os
import sys
from pathlib import Path

# Suppress external library logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['ESSENTIA_LOG_LEVEL'] = 'error'
os.environ['MUSICEXTRACTOR_LOG_LEVEL'] = 'error'
os.environ['TENSORFLOW_LOG_LEVEL'] = '2'
os.environ['LIBROSA_LOG_LEVEL'] = 'error'

# Add src to path for imports
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from cli.main import main as cli_main


def main():
    """Main entry point - routes to CLI or API based on arguments."""
    # Check if running as CLI
    if len(sys.argv) > 1:
        # CLI mode
        return cli_main()
    else:
        # API mode - import and run FastAPI
        try:
            import uvicorn
            from api.main import app
            
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info"
            )
        except ImportError:
            print("API dependencies not available. Running in CLI mode only.")
            return cli_main()


if __name__ == "__main__":
    sys.exit(main()) 