#!/usr/bin/env python3
"""
Script to remove all emojis from the playlist-generator-simple project.
This script will scan all Python files and remove emoji characters.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Set

# Unicode ranges for emojis
EMOJI_PATTERNS = [
    # Emoji ranges
    r'[\U0001F600-\U0001F64F]',  # Emoticons
    r'[\U0001F300-\U0001F5FF]',  # Misc Symbols and Pictographs
    r'[\U0001F680-\U0001F6FF]',  # Transport and Map Symbols
    r'[\U0001F1E0-\U0001F1FF]',  # Regional Indicator Symbols
    r'[\U00002600-\U000027BF]',  # Misc Symbols
    r'[\U0001F900-\U0001F9FF]',  # Supplemental Symbols and Pictographs
    r'[\U0001F018-\U0001F270]',  # Misc Symbols and Arrows
    r'[\U0001F600-\U0001F636]',  # Emoticons
    r'[\U0001F645-\U0001F64F]',  # Emoticons
    r'[\U0001F680-\U0001F6C5]',  # Transport and Map Symbols
    r'[\U0001F6CB-\U0001F6D2]',  # Transport and Map Symbols
    r'[\U0001F6E0-\U0001F6EC]',  # Transport and Map Symbols
    r'[\U0001F6F4-\U0001F6F9]',  # Transport and Map Symbols
    r'[\U0001F910-\U0001F93A]',  # Supplemental Symbols and Pictographs
    r'[\U0001F93C-\U0001F93E]',  # Supplemental Symbols and Pictographs
    r'[\U0001F940-\U0001F970]',  # Supplemental Symbols and Pictographs
    r'[\U0001F973-\U0001F976]',  # Supplemental Symbols and Pictographs
    r'[\U0001F97A]',             # Supplemental Symbols and Pictographs
    r'[\U0001F97C-\U0001F9A2]',  # Supplemental Symbols and Pictographs
    r'[\U0001F9B0-\U0001F9B9]',  # Supplemental Symbols and Pictographs
    r'[\U0001F9C0-\U0001F9C2]',  # Supplemental Symbols and Pictographs
    r'[\U0001F9D0-\U0001F9FF]',  # Supplemental Symbols and Pictographs
    r'[\U0001FA70-\U0001FA73]',  # Symbols and Pictographs Extended-A
    r'[\U0001FA78-\U0001FA7A]',  # Symbols and Pictographs Extended-A
    r'[\U0001FA7C-\U0001FA7F]',  # Symbols and Pictographs Extended-A
    r'[\U0001FA80-\U0001FA82]',  # Symbols and Pictographs Extended-A
    r'[\U0001FA90-\U0001FA95]',  # Symbols and Pictographs Extended-A
    r'[\U0001FA96-\U0001FAA8]',  # Symbols and Pictographs Extended-A
    r'[\U0001FAB0-\U0001FAB6]',  # Symbols and Pictographs Extended-A
    r'[\U0001FAC0-\U0001FAC2]',  # Symbols and Pictographs Extended-A
    r'[\U0001FAD0-\U0001FAD6]',  # Symbols and Pictographs Extended-A
]

# Combined emoji regex pattern
EMOJI_REGEX = re.compile('|'.join(EMOJI_PATTERNS))

def is_emoji(char: str) -> bool:
    """Check if a character is an emoji."""
    return bool(EMOJI_REGEX.match(char))

def remove_emojis_from_text(text: str) -> str:
    """Remove all emoji characters from text."""
    return EMOJI_REGEX.sub('', text)

def get_python_files(directory: str) -> List[Path]:
    """Get all Python files in the directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common directories that shouldn't be modified
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files

def process_file(file_path: Path) -> tuple[bool, int, Set[str]]:
    """
    Process a single file to remove emojis.
    
    Returns:
        (modified, emoji_count, emojis_found)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all emojis in the content
        emojis_found = set(EMOJI_REGEX.findall(content))
        emoji_count = len(emojis_found)
        
        if emoji_count == 0:
            return False, 0, set()
        
        # Remove emojis
        cleaned_content = remove_emojis_from_text(content)
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True, emoji_count, emojis_found
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, 0, set()

def main():
    """Main function to remove emojis from all Python files."""
    # Get the playlist-generator-simple directory
    script_dir = Path(__file__).parent
    simple_dir = script_dir / 'playlist-generator-simple'
    
    if not simple_dir.exists():
        print(f"Error: {simple_dir} does not exist")
        print("Make sure this script is in the same directory as playlist-generator-simple")
        sys.exit(1)
    
    print(f"Scanning for Python files in: {simple_dir}")
    
    # Get all Python files
    python_files = get_python_files(str(simple_dir))
    
    if not python_files:
        print("No Python files found")
        return
    
    print(f"Found {len(python_files)} Python files")
    
    # Process each file
    total_modified = 0
    total_emojis = 0
    all_emojis = set()
    
    for file_path in python_files:
        modified, emoji_count, emojis_found = process_file(file_path)
        
        if modified:
            total_modified += 1
            total_emojis += emoji_count
            all_emojis.update(emojis_found)
            print(f"Modified: {file_path} ({emoji_count} emojis removed)")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Files processed: {len(python_files)}")
    print(f"Files modified: {total_modified}")
    print(f"Total emojis removed: {total_emojis}")
    
    if all_emojis:
        print(f"Emojis found: {', '.join(sorted(all_emojis))}")
    
    if total_modified > 0:
        print(f"\n✅ Successfully removed emojis from {total_modified} files")
    else:
        print("\nℹ️  No emojis found in any files")

if __name__ == "__main__":
    main() 