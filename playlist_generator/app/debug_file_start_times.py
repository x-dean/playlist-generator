#!/usr/bin/env python3
"""
Debug script to identify file_start_times variable issues.
"""

import os
import sys

def check_file_for_issues():
    """Check the analysis_manager.py file for potential issues."""
    file_path = "music_analyzer/analysis_manager.py"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"🔍 Checking file: {file_path}")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    issues_found = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        
        # Check for problematic assignments
        if "file_start_times = time.time()" in line:
            issues_found.append(f"Line {i}: Direct assignment to time.time()")
        
        # Check for variable usage before declaration
        if "file_start_times[" in line and i < 350:
            # Look for declaration before this line
            declared_before = False
            for j in range(i-1, 0, -1):
                if "file_start_times = {}" in lines[j]:
                    declared_before = True
                    break
            if not declared_before:
                issues_found.append(f"Line {i}: Usage before declaration")
        
        # Check for incorrect assignments
        if "file_start_times = " in line and "time.time()" in line:
            issues_found.append(f"Line {i}: Incorrect assignment pattern")
    
    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"   {issue}")
    else:
        print("✅ No obvious issues found in the file")
    
    # Show the relevant lines around the error location
    print(f"\n📋 Lines around line 351:")
    start_line = max(1, 346)
    end_line = min(len(lines), 356)
    
    for i in range(start_line, end_line + 1):
        marker = ">>> " if i == 351 else "    "
        print(f"{marker}{i:3d}: {lines[i-1].rstrip()}")

def check_docker_file():
    """Check if there's a different version in Docker."""
    print("\n🐳 Checking Docker container file...")
    print("If you're running in Docker, the container might have an older version.")
    print("Try rebuilding the Docker container:")
    print("   docker-compose down")
    print("   docker-compose build --no-cache")
    print("   docker-compose up")

def main():
    """Main function."""
    print("🔧 Debug: file_start_times variable issue")
    print("=" * 50)
    
    check_file_for_issues()
    check_docker_file()
    
    print("\n💡 Recommendations:")
    print("1. Rebuild Docker container if using Docker")
    print("2. Clear Python cache: find . -name '*.pyc' -delete")
    print("3. Restart the application")

if __name__ == "__main__":
    main() 