#!/bin/bash

# Script to copy 100 audio files with specific size distribution
# 30 files over 50MB and 70 files under 50MB

SOURCE_DIR="/root/music/library/"
DEST_DIR="/tmp/"
TEMP_DIR="/tmp/audio_copy_temp/"

# Create temporary directory
mkdir -p "$TEMP_DIR"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist!"
    echo "Please create the directory and add some audio files first."
    exit 1
fi

echo "Starting audio file copy process..."
echo "Source directory: $SOURCE_DIR"

# Find all audio files in source directory (using null-terminated output for proper path handling)
find "$SOURCE_DIR" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.aac" \) -print0 | tr '\0' '\n' > "$TEMP_DIR/all_audio_files.txt"

total_files=$(wc -l < "$TEMP_DIR/all_audio_files.txt")
echo "Found $total_files total audio files"

if [ "$total_files" -eq 0 ]; then
    echo "Error: No audio files found in '$SOURCE_DIR'"
    echo "Please add some audio files (.mp3, .flac, .wav, .m4a, .ogg, .aac) to the source directory."
    exit 1
fi

# Separate files by size using a more compatible approach
echo "Analyzing file sizes..."

# Create a temporary script to check file sizes with proper path handling
cat > "$TEMP_DIR/check_sizes.sh" << 'EOF'
#!/bin/bash
source_dir="$1"
temp_dir="$2"

# Files over 50MB - use null-terminated output to handle spaces in filenames
find "$source_dir" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.aac" \) -print0 | while IFS= read -r -d $'\0' file; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    if [ "$size" -gt 52428800 ]; then
        echo "$file" >> "$temp_dir/large_files.txt"
    fi
done

# Files under 50MB - use null-terminated output to handle spaces in filenames
find "$source_dir" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.aac" \) -print0 | while IFS= read -r -d $'\0' file; do
    size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null)
    if [ "$size" -le 52428800 ]; then
        echo "$file" >> "$temp_dir/small_files.txt"
    fi
done
EOF

chmod +x "$TEMP_DIR/check_sizes.sh"
"$TEMP_DIR/check_sizes.sh" "$SOURCE_DIR" "$TEMP_DIR"

# Count available files
large_count=$(wc -l < "$TEMP_DIR/large_files.txt")
small_count=$(wc -l < "$TEMP_DIR/small_files.txt")

echo "Found $large_count files over 50MB"
echo "Found $small_count files under 50MB"

# Check if we have enough files
if [ "$large_count" -lt 30 ]; then
    echo "Warning: Only $large_count files over 50MB available, will copy all of them"
    files_to_copy_large=$large_count
else
    files_to_copy_large=30
fi

if [ "$small_count" -lt 70 ]; then
    echo "Warning: Only $small_count files under 50MB available, will copy all of them"
    files_to_copy_small=$small_count
else
    files_to_copy_small=70
fi

# Shuffle and select files
echo "Selecting files to copy..."

# Select large files (over 50MB)
shuf "$TEMP_DIR/large_files.txt" | head -n "$files_to_copy_large" > "$TEMP_DIR/selected_large.txt"

# Select small files (under 50MB)
shuf "$TEMP_DIR/small_files.txt" | head -n "$files_to_copy_small" > "$TEMP_DIR/selected_small.txt"

# Combine selected files
cat "$TEMP_DIR/selected_large.txt" "$TEMP_DIR/selected_small.txt" > "$TEMP_DIR/files_to_copy.txt"

total_selected=$(wc -l < "$TEMP_DIR/files_to_copy.txt")
echo "Selected $total_selected files to copy"

# Copy files
echo "Copying files to $DEST_DIR..."

copied_count=0
failed_count=0

while IFS= read -r file; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        # Create a unique name if file already exists
        counter=1
        dest_file="$DEST_DIR$filename"
        while [ -f "$dest_file" ]; do
            name_without_ext="${filename%.*}"
            ext="${filename##*.}"
            dest_file="$DEST_DIR${name_without_ext}_${counter}.$ext"
            ((counter++))
        done
        
        if cp "$file" "$dest_file"; then
            echo "Copied: $filename -> $(basename "$dest_file")"
            ((copied_count++))
        else
            echo "Failed to copy: $filename"
            ((failed_count++))
        fi
    else
        echo "File not found: $file"
        ((failed_count++))
    fi
done < "$TEMP_DIR/files_to_copy.txt"

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "Copy operation completed!"
echo "Successfully copied: $copied_count files"
echo "Failed to copy: $failed_count files"
echo "Total files processed: $total_selected"

# Show size distribution of copied files
echo ""
echo "Size distribution of copied files:"
find "$DEST_DIR" -type f \( -iname "*.mp3" -o -iname "*.flac" -o -iname "*.wav" -o -iname "*.m4a" -o -iname "*.ogg" -o -iname "*.aac" \) -exec ls -lh {} \; | awk '{print $5, $9}' | sort -h 