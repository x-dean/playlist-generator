#!/usr/bin/env python3
"""
Quick test for domain entities.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("🚀 Quick Domain Entities Test")
    print("=" * 40)
    
    try:
        # Test imports
        from domain.entities import AudioFile, FeatureSet, Metadata, AnalysisResult, Playlist
        print("✅ All domain entities imported successfully")
        
        # Test AudioFile creation
        test_path = Path("/music/test.mp3")
        audio_file = AudioFile(file_path=test_path)
        print(f"✅ AudioFile created: {audio_file.file_name}")
        
        # Test FeatureSet creation
        feature_set = FeatureSet(audio_file_id=audio_file.id, bpm=120.0)
        print(f"✅ FeatureSet created: BPM = {feature_set.bpm}")
        
        # Test Metadata creation
        metadata = Metadata(audio_file_id=audio_file.id, title="Test Song")
        print(f"✅ Metadata created: {metadata.display_title}")
        
        # Test AnalysisResult creation
        analysis_result = AnalysisResult(audio_file=audio_file)
        print(f"✅ AnalysisResult created: {analysis_result.file_name}")
        
        # Test Playlist creation
        playlist = Playlist(name="Test Playlist")
        print(f"✅ Playlist created: {playlist.name}")
        
        print("\n🎉 All domain entities working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 