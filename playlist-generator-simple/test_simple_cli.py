#!/usr/bin/env python3
"""
Simple test to debug CLI output issues.
"""

import subprocess
import sys

def test_command(cmd, description):
    print(f"\n🔧 Testing: {description}")
    print(f"📝 Command: {cmd}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(f"✅ Exit code: {result.returncode}")
        
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️ Errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🧪 Simple CLI Test")
    print("=" * 60)
    
    # Test basic commands
    test_command("python src/enhanced_cli.py --help", "Help command")
    test_command("python src/enhanced_cli.py stats", "Stats command")
    test_command("python src/enhanced_cli.py playlist-methods", "Playlist methods")
    test_command("python src/enhanced_cli.py config", "Config command")

if __name__ == "__main__":
    main() 