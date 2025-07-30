#!/usr/bin/env python3
"""
Test script for Docker setup verification.
This script checks if the Docker environment is properly configured.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def check_docker_installation():
    """Check if Docker and Docker Compose are installed."""
    print("🔍 Checking Docker installation...")
    
    try:
        # Check Docker
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker: {result.stdout.strip()}")
        
        # Check Docker Compose
        result = subprocess.run(['docker-compose', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Docker Compose: {result.stdout.strip()}")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Docker check failed: {e}")
        return False
    except FileNotFoundError:
        print("❌ Docker or Docker Compose not found. Please install Docker.")
        return False


def check_required_files():
    """Check if required files exist."""
    print("\n📁 Checking required files...")
    
    required_files = [
        'Dockerfile',
        'docker-compose.yml',
        'requirements.txt',
        'playlista.conf',
        'src/'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        return False
    
    return True


def check_directory_structure():
    """Check if required directories exist or can be created."""
    print("\n📂 Checking directory structure...")
    
    required_dirs = [
        'music',
        'playlists', 
        'cache',
        'logs',
        'failed_files',
        'models'
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ {dir_name}/ (created)")
            except Exception as e:
                print(f"❌ {dir_name}/ - Cannot create: {e}")
                return False
    
    return True


def test_docker_build():
    """Test Docker build process."""
    print("\n🔨 Testing Docker build...")
    
    try:
        # Build the image
        print("Building Docker image...")
        result = subprocess.run(
            ['docker-compose', 'build'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print("✅ Docker build successful")
            return True
        else:
            print(f"❌ Docker build failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Docker build timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Docker build error: {e}")
        return False


def test_container_startup():
    """Test if container can start and run basic commands."""
    print("\n🚀 Testing container startup...")
    
    try:
        # Test container startup
        result = subprocess.run(
            ['docker-compose', 'run', '--rm', 'playlista-simple', 'python', '-c', 
             'import sys; print("Container startup test successful")'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ Container startup successful")
            print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Container startup failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Container startup timed out")
        return False
    except Exception as e:
        print(f"❌ Container startup error: {e}")
        return False


def test_cli_help():
    """Test if the CLI help command works."""
    print("\n💬 Testing CLI help command...")
    
    try:
        result = subprocess.run(
            ['docker-compose', 'run', '--rm', 'playlista-simple', 'playlista', '--help'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ CLI help command successful")
            # Show first few lines of help
            help_lines = result.stdout.strip().split('\n')[:5]
            for line in help_lines:
                print(f"   {line}")
            return True
        else:
            print(f"❌ CLI help command failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ CLI help command timed out")
        return False
    except Exception as e:
        print(f"❌ CLI help command error: {e}")
        return False


def create_sample_music():
    """Create a sample music file for testing."""
    print("\n🎵 Creating sample music file for testing...")
    
    music_dir = Path('music')
    music_dir.mkdir(exist_ok=True)
    
    # Create a simple test file (not actual audio, just for testing)
    test_file = music_dir / 'test_song.mp3'
    
    try:
        # Create a dummy file for testing
        with open(test_file, 'w') as f:
            f.write("# This is a test file for Docker testing\n")
            f.write("# Replace with actual music files for real testing\n")
        
        print(f"✅ Created test file: {test_file}")
        print("⚠️ Replace with actual music files for real testing")
        return True
    except Exception as e:
        print(f"❌ Failed to create test file: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Docker Setup Test")
    print("=" * 50)
    
    tests = [
        ("Docker Installation", check_docker_installation),
        ("Required Files", check_required_files),
        ("Directory Structure", check_directory_structure),
        ("Sample Music", create_sample_music),
        ("Docker Build", test_docker_build),
        ("Container Startup", test_container_startup),
        ("CLI Help", test_cli_help),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Docker setup is ready for use.")
        print("\n📋 Next steps:")
        print("   1. Add your music files to the 'music/' directory")
        print("   2. Run: docker-compose run --rm playlista-simple analyze")
        print("   3. Run: docker-compose run --rm playlista-simple playlist")
        print("   4. Check generated playlists in 'playlists/' directory")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 