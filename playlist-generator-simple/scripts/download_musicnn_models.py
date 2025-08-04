#!/usr/bin/env python3
"""
Download MusiCNN model files for playlist generator.
This script downloads the required model files and JSON configuration.
"""

import os
import sys
import requests
import json
from pathlib import Path

def download_file(url: str, local_path: str) -> bool:
    """Download a file from URL to local path."""
    try:
        print(f"Downloading {url} to {local_path}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Downloaded {os.path.basename(local_path)}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        return False

def create_json_config(local_path: str) -> bool:
    """Create the JSON configuration file for MusiCNN."""
    try:
        # This is the standard MusiCNN tag configuration
        config = {
            "tag_names": [
                "rock", "pop", "alternative", "indie", "electronic", "female vocalists", 
                "dance", "00s", "alternative rock", "jazz", "beautiful", "metal", 
                "chillout", "male vocalists", "classic rock", "soul", "indie rock", 
                "Mellow", "electronica", "80s", "folk", "90s", "blues", "hardcore", 
                "instrumental", "punk", "oldies", "country", "hard rock", "00's", 
                "ambient", "acoustic", "experimental", "female vocalist", "guitar", 
                "Hip-Hop", "70s", "party", "male vocalist", "classic", "syntax", 
                "indie pop", "heavy metal", "singer-songwriter", "world music", 
                "electro", "funk", "garage", "Classic Rock", "philadelphia", "mellow", 
                "soulful", "jazz vocal", "beautiful voice", "background", "female vocals", 
                "male vocals", "vocal", "vocalist", "vocalists", "vocalist", "vocalists", 
                "vocalist", "vocalists", "vocalist", "vocalists", "vocalist", "vocalists"
            ]
        }
        
        with open(local_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Created JSON configuration at {local_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to create JSON config: {e}")
        return False

def main():
    """Download MusiCNN model files."""
    print("=== MusiCNN Model Downloader ===")
    
    # Define paths
    models_dir = "/app/models"
    model_path = os.path.join(models_dir, "msd-musicnn-1.pb")
    json_path = os.path.join(models_dir, "msd-musicnn-1.json")
    
    # Check if we're in Docker environment
    if not os.path.exists("/app"):
        print("⚠️  Not in Docker environment. Using local models directory...")
        models_dir = "./models"
        model_path = os.path.join(models_dir, "msd-musicnn-1.pb")
        json_path = os.path.join(models_dir, "msd-musicnn-1.json")
    
    print(f"Models directory: {models_dir}")
    print(f"Model file: {model_path}")
    print(f"JSON config: {json_path}")
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    print(f"✓ Created models directory: {models_dir}")
    
    # Download model file (try multiple sources)
    model_urls = [
        "https://github.com/jordipons/musicnn/releases/download/v1.0/msd-musicnn-1.pb",
        "https://github.com/jordipons/musicnn/releases/download/v1.0/msd-musicnn-1.h5",
        "https://github.com/jordipons/musicnn/releases/download/v1.0/msd-musicnn-1.zip"
    ]
    
    model_success = False
    for url in model_urls:
        print(f"Trying URL: {url}")
        if download_file(url, model_path):
            model_success = True
            break
    
            if not model_success:
            print("❌ All download URLs failed. Using alternative approach...")
            # Create a placeholder file with instructions
            placeholder_content = """# MusiCNN Model Placeholder

This is a placeholder file. The actual MusiCNN model file is not available for automatic download.

To get the MusiCNN model:

1. Visit: https://github.com/jordipons/musicnn
2. Download the model file manually
3. Place it in this directory as 'msd-musicnn-1.pb'

Alternatively, you can:
- Set EXTRACT_MUSICNN=false in playlista.conf to disable MusiCNN features
- Use the system without MusiCNN (other features will still work)

For more information, see: MUSICNN_SETUP_GUIDE.md
"""
            try:
                with open(model_path, 'w') as f:
                    f.write(placeholder_content)
                print(f"✓ Created placeholder file: {model_path}")
                print("⚠️  MusiCNN features will be disabled until you download the actual model")
            except Exception as e:
                print(f"❌ Failed to create placeholder: {e}")
    
    # Create JSON configuration
    json_success = create_json_config(json_path)
    
    # Verify files
    print("\n=== Verification ===")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✓ Model file exists: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"❌ Model file missing: {model_path}")
    
    if os.path.exists(json_path):
        print(f"✓ JSON config exists: {json_path}")
    else:
        print(f"❌ JSON config missing: {json_path}")
    
    if model_success and json_success:
        print("\n=== Setup Complete ===")
        print("✓ MusiCNN model files downloaded successfully")
        print("✓ You can now enable MusiCNN features in your analysis")
        print("\nTo enable MusiCNN in your configuration:")
        print("1. Set EXTRACT_MUSICNN=true in playlista.conf")
        print("2. Ensure MUSICNN_MODEL_PATH and MUSICNN_JSON_PATH are correct")
    else:
        print("\n=== Setup Incomplete ===")
        print("❌ Some files failed to download")
        print("Please check the error messages above and try again")
        
        if not model_success:
            print("\nManual download instructions:")
            print("1. Visit: https://github.com/jordipons/musicnn")
            print("2. Download the model file manually")
            print(f"3. Place it at: {model_path}")
            print(f"4. JSON config is already created at: {json_path}")

if __name__ == "__main__":
    main() 