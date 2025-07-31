#!/usr/bin/env python3
"""
Script to remove emojis from external APIs logging messages.
"""

import re
import os
from pathlib import Path

def remove_emojis_from_file(file_path):
    """Remove emojis from a file."""
    print(f"Processing: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Original content for comparison
    original_content = content
    
    # Common emoji patterns found in the logging messages
    emoji_patterns = [
        # Music and search emojis
        r'🎵\s*',  # Music note
        r'🔍\s*',  # Magnifying glass
        r'📝\s*',  # Memo
        r'🎯\s*',  # Target
        r'✅\s*',  # Check mark
        r'❌\s*',  # Cross mark
        r'⚠️\s*',  # Warning
        r'️\s*',   # Invisible character + space
    ]
    
    # Remove emojis
    for pattern in emoji_patterns:
        content = re.sub(pattern, '', content)
    
    # Clean up extra spaces that might have been left
    content = re.sub(r'f"(\s+)([^"]*)"', r'f"\2"', content)
    content = re.sub(r'f"([^"]*)(\s+)"', r'f"\1"', content)
    
    # Check if any changes were made
    if content != original_content:
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Removed emojis from {file_path}")
        return True
    else:
        print(f"  No emojis found in {file_path}")
        return False

def main():
    """Main function to process external APIs files."""
    print("=" * 60)
    print("EMOJI REMOVAL SCRIPT")
    print("=" * 60)
    
    # Files to process - using relative paths from current directory
    files_to_process = [
        "../src/core/external_apis.py",
        "src/core/external_apis.py"
    ]
    
    total_changes = 0
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            if remove_emojis_from_file(file_path):
                total_changes += 1
        else:
            print(f"  File not found: {file_path}")
    
    print("\n" + "=" * 60)
    print(f"SUMMARY: Processed {len(files_to_process)} files, made changes to {total_changes} files")
    print("=" * 60)

if __name__ == "__main__":
    main() 