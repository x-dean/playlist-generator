#!/usr/bin/env python3
"""
Production-ready analysis workflow for Playlista v2
This script analyzes music files and stores results in the database for the UI
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

logger = get_logger("production_analysis")

class ProductionAnalysisEngine:
    """Production analysis engine that stores results for the UI"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.processed_count = 0
        self.failed_count = 0
        
    async def analyze_music_library(self, limit: int = None):
        """Analyze the entire music library and store results"""
        
        print("🎵 PLAYLISTA V2 PRODUCTION ANALYSIS")
        print("=" * 50)
        
        # Initialize ML models
        print("🤖 Loading ML models...")
        await model_manager.load_models()
        print("✅ ML models loaded successfully")
        
        # Scan for music files
        music_dir = "/music"
        print(f"📁 Scanning {music_dir} for audio files...")
        
        audio_files = []
        for root, dirs, files in os.walk(music_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac')):
                    audio_files.append(os.path.join(root, file))
        
        print(f"🎯 Found {len(audio_files)} audio files")
        
        if not audio_files:
            print("❌ No audio files found in the music directory")
            return
        
        # Apply limit if specified
        if limit:
            audio_files = audio_files[:limit]
            print(f"📋 Processing first {len(audio_files)} files (limited)")
        
        # Check existing tracks in database
        existing_files = await self._get_existing_tracks()
        new_files = [f for f in audio_files if f not in existing_files]
        
        print(f"📊 Analysis status:")
        print(f"   - Total files: {len(audio_files)}")
        print(f"   - Already in DB: {len(existing_files)}")
        print(f"   - New files: {len(new_files)}")
        
        # Process new files
        if new_files:
            print(f"\n🔬 Starting analysis of {len(new_files)} new files...")
            print("=" * 50)
            
            for i, file_path in enumerate(new_files, 1):
                print(f"\n[{i}/{len(new_files)}] Analyzing: {os.path.basename(file_path)}")
                
                try:
                    await self._analyze_and_store_track(file_path)
                    self.processed_count += 1
                    print(f"✅ Success ({self.processed_count} total)")
                    
                except Exception as e:
                    self.failed_count += 1
                    print(f"❌ Failed: {e}")
                    logger.error(f"Analysis failed for {file_path}: {e}")
        
        # Final summary
        print(f"\n🎯 ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"✅ Successfully processed: {self.processed_count}")
        print(f"❌ Failed: {self.failed_count}")
        print(f"📊 Total in database: {len(existing_files) + self.processed_count}")
        
        # Show database status
        await self._show_database_status()
        
    async def _get_existing_tracks(self):
        """Get list of files already in database"""
        try:
            async with get_db_session() as session:
                from sqlalchemy import text
                result = await session.execute(text("SELECT file_path FROM tracks"))
                return {row[0] for row in result.fetchall()}
        except:
            return set()
    
    async def _analyze_and_store_track(self, file_path: str):
        """Analyze single track and store in database"""
        
        # Extract basic metadata
        metadata = self._extract_metadata(file_path)
        file_info = self._get_file_info(file_path)
        
        print(f"   🎭 {metadata.get('title', 'Unknown')} by {metadata.get('artist', 'Unknown')}")
        
        # Extract audio features
        start_time = time.time()
        audio_features = await self.feature_extractor.extract_comprehensive_features(file_path)
        analysis_time = time.time() - start_time
        
        print(f"   🔬 Features extracted in {analysis_time:.1f}s")
        
        # Store in database
        track_id = str(uuid.uuid4())
        
        async with get_db_session() as session:
            track = Track(
                id=track_id,
                file_path=file_path,
                filename=os.path.basename(file_path),
                title=metadata.get('title'),
                artist=metadata.get('artist'),
                album=metadata.get('album'),
                duration=metadata.get('duration', 0),
                file_size=file_info['size_bytes'],
                audio_features=audio_features
            )
            
            session.add(track)
            await session.commit()
            
        print(f"   💾 Stored in database: {track_id[:8]}...")
    
    def _extract_metadata(self, file_path: str) -> dict:
        """Extract metadata using Mutagen"""
        try:
            file = mutagen.File(file_path)
            if not file:
                return {}
            
            metadata = {}
            
            # Extract common metadata
            if hasattr(file, 'tags') and file.tags:
                tags = file.tags
                
                # Handle different tag formats
                title_keys = ['TIT2', 'TITLE', '\xa9nam']
                artist_keys = ['TPE1', 'ARTIST', '\xa9ART'] 
                album_keys = ['TALB', 'ALBUM', '\xa9alb']
                
                metadata['title'] = self._get_first_value(tags, title_keys)
                metadata['artist'] = self._get_first_value(tags, artist_keys)
                metadata['album'] = self._get_first_value(tags, album_keys)
            
            # Audio properties
            if hasattr(file, 'info') and file.info:
                metadata['duration'] = getattr(file.info, 'length', 0)
                metadata['bitrate'] = getattr(file.info, 'bitrate', 0)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {file_path}: {e}")
            return {}
    
    def _get_first_value(self, tags, keys):
        """Get first available value from tag keys"""
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, list) and value:
                    return str(value[0])
                elif value:
                    return str(value)
        return None
    
    def _get_file_info(self, file_path: str) -> dict:
        """Get basic file information"""
        stat = os.stat(file_path)
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
        }
    
    async def _show_database_status(self):
        """Show current database status"""
        try:
            async with get_db_session() as session:
                from sqlalchemy import text
                
                # Count tracks
                result = await session.execute(text("SELECT COUNT(*) FROM tracks"))
                total_tracks = result.scalar() or 0
                
                # Count analyzed tracks
                result = await session.execute(text("SELECT COUNT(*) FROM tracks WHERE audio_features IS NOT NULL"))
                analyzed_tracks = result.scalar() or 0
                
                # Recent tracks
                result = await session.execute(text("""
                    SELECT title, artist, created_at 
                    FROM tracks 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """))
                recent_tracks = result.fetchall()
                
                print(f"\n📊 DATABASE STATUS:")
                print(f"   📄 Total tracks: {total_tracks}")
                print(f"   ✅ Analyzed tracks: {analyzed_tracks}")
                print(f"   📊 Analysis coverage: {(analyzed_tracks/total_tracks*100):.1f}%" if total_tracks > 0 else "   📊 Analysis coverage: 0%")
                
                if recent_tracks:
                    print(f"\n📋 Recent tracks:")
                    for track in recent_tracks:
                        title = track[0] or 'Unknown'
                        artist = track[1] or 'Unknown'
                        print(f"   🎵 {title} by {artist}")
                
        except Exception as e:
            print(f"❌ Could not get database status: {e}")

async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze music library for Playlista v2')
    parser.add_argument('--limit', type=int, help='Limit number of files to process')
    parser.add_argument('--quick', action='store_true', help='Quick test with 5 files')
    
    # Parse args from sys.argv if available
    try:
        args = parser.parse_args()
    except:
        # If argument parsing fails, use defaults
        args = type('Args', (), {'limit': None, 'quick': False})()
    
    # Set limit based on flags
    limit = None
    if hasattr(args, 'quick') and args.quick:
        limit = 5
    elif hasattr(args, 'limit') and args.limit:
        limit = args.limit
    
    # Run analysis
    engine = ProductionAnalysisEngine()
    await engine.analyze_music_library(limit=limit)

if __name__ == "__main__":
    asyncio.run(main())
