#!/usr/bin/env python3
"""
Fixed analysis script that properly stores data in the database
"""

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.append('/app')

from app.analysis.features import FeatureExtractor
from app.analysis.models import model_manager
from app.database.models import Track
from app.database.connection import get_db_session
from app.core.logging import get_logger
import mutagen

logger = get_logger("fixed_analysis")

class FixedAnalysisEngine:
    """Fixed analysis engine for database storage"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.processed_count = 0
        self.failed_count = 0
        
    async def analyze_tracks(self, limit: int = 5):
        """Analyze tracks and store in database"""
        
        print("🎵 FIXED ANALYSIS ENGINE")
        print("=" * 40)
        
        # Load models
        print("🤖 Loading ML models...")
        await model_manager.load_models()
        print("✅ Models loaded")
        
        # Find audio files
        music_dir = "/music"
        audio_files = []
        
        for root, dirs, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    audio_files.append(os.path.join(root, file))
        
        # Limit files
        audio_files = audio_files[:limit]
        print(f"📁 Processing {len(audio_files)} files")
        
        # Process each file
        for i, file_path in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] {os.path.basename(file_path)}")
            
            try:
                await self._process_single_track(file_path)
                self.processed_count += 1
                print(f"✅ Success")
                
            except Exception as e:
                self.failed_count += 1
                print(f"❌ Failed: {e}")
        
        print(f"\n🎯 Complete: {self.processed_count} success, {self.failed_count} failed")
        
        # Show database status
        await self._check_database()
    
    async def _process_single_track(self, file_path: str):
        """Process a single track"""
        
        # Extract metadata
        metadata = self._extract_metadata(file_path)
        file_info = self._get_file_info(file_path)
        
        # Extract features
        start_time = time.time()
        features = await self.feature_extractor.extract_comprehensive_features(file_path)
        analysis_time = time.time() - start_time
        
        print(f"   🔬 Analysis: {analysis_time:.1f}s")
        
        # Store in database
        track_id = str(uuid.uuid4())
        
        # Create session manually
        from app.database.connection import async_session_maker
        
        async with async_session_maker() as session:
            track = Track(
                id=track_id,
                file_path=file_path,
                filename=os.path.basename(file_path),
                title=metadata.get('title'),
                artist=metadata.get('artist'),
                album=metadata.get('album'),
                duration=metadata.get('duration', 0),
                file_size=file_info['size_bytes'],
                audio_features=features
            )
            
            session.add(track)
            await session.commit()
            
        print(f"   💾 Stored: {track_id[:8]}...")
    
    def _extract_metadata(self, file_path: str) -> dict:
        """Extract basic metadata"""
        try:
            file = mutagen.File(file_path)
            if not file:
                return {}
            
            metadata = {}
            
            if hasattr(file, 'tags') and file.tags:
                tags = file.tags
                
                # Common tag mappings
                title_keys = ['TIT2', 'TITLE', '\xa9nam']
                artist_keys = ['TPE1', 'ARTIST', '\xa9ART'] 
                album_keys = ['TALB', 'ALBUM', '\xa9alb']
                
                metadata['title'] = self._get_first_value(tags, title_keys)
                metadata['artist'] = self._get_first_value(tags, artist_keys)
                metadata['album'] = self._get_first_value(tags, album_keys)
            
            if hasattr(file, 'info') and file.info:
                metadata['duration'] = getattr(file.info, 'length', 0)
            
            return metadata
            
        except Exception:
            return {}
    
    def _get_first_value(self, tags, keys):
        """Get first available tag value"""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif value:
                    return str(value)
        return None
    
    def _get_file_info(self, file_path: str) -> dict:
        """Get file information"""
        stat = os.stat(file_path)
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
        }
    
    async def _check_database(self):
        """Check database contents"""
        try:
            from app.database.connection import async_session_maker
            from sqlalchemy import text
            
            async with async_session_maker() as session:
                # Count tracks
                result = await session.execute(text("SELECT COUNT(*) FROM tracks"))
                count = result.scalar()
                
                print(f"\n📊 Database: {count} tracks total")
                
                if count > 0:
                    # Show recent tracks
                    result = await session.execute(text("""
                        SELECT title, artist, filename
                        FROM tracks 
                        ORDER BY created_at DESC 
                        LIMIT 3
                    """))
                    tracks = result.fetchall()
                    
                    print("📋 Recent tracks:")
                    for track in tracks:
                        title = track[0] or 'Unknown'
                        artist = track[1] or 'Unknown'
                        filename = track[2]
                        print(f"   🎵 {title} by {artist} ({filename})")
                        
        except Exception as e:
            print(f"❌ Database check failed: {e}")

async def main():
    """Main function"""
    engine = FixedAnalysisEngine()
    await engine.analyze_tracks(limit=5)

if __name__ == "__main__":
    asyncio.run(main())
