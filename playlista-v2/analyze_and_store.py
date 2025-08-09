#!/usr/bin/env python3
"""
Complete analysis workflow that stores results in the database
This shows exactly how analysis works and where data is stored
"""

import asyncio
import os
import sys
import time
import json
import uuid
from pathlib import Path

sys.path.append('/app')

from app.analysis.features import FeatureExtractor
from app.analysis.models import model_manager
from app.database.models import Track, AnalysisJob
from app.database.connection import get_db_session
from app.core.logging import get_logger
from app.core.external_apis import external_api_manager
import mutagen

logger = get_logger("analyze_and_store")

class DatabaseAnalysisWorkflow:
    """Complete analysis workflow that stores results in PostgreSQL"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.analyzed_count = 0
        
    async def analyze_and_store_track(self, file_path: str) -> dict:
        """Analyze a single track and store results in database"""
        
        print(f"\n🎵 Analyzing: {os.path.basename(file_path)}")
        
        try:
            # 1. Extract basic file info
            file_info = self._get_file_info(file_path)
            print(f"   📁 File: {file_info['size_mb']:.1f}MB")
            
            # 2. Extract metadata using Mutagen
            metadata = self._extract_metadata(file_path)
            print(f"   🎭 Metadata: {metadata.get('title', 'Unknown')} by {metadata.get('artist', 'Unknown')}")
            
            # 3. Extract audio features
            print(f"   🔬 Extracting audio features...")
            start_time = time.time()
            audio_features = await self.feature_extractor.extract_comprehensive_features(file_path)
            analysis_time = time.time() - start_time
            print(f"   ✅ Features extracted in {analysis_time:.1f}s")
            
            # 4. Get external API data (simulated for now)
            external_data = await self._get_external_data(metadata)
            
            # 5. Store in database
            track_data = await self._store_in_database(file_path, file_info, metadata, audio_features, external_data)
            
            print(f"   💾 Stored in database with ID: {track_data['id']}")
            
            self.analyzed_count += 1
            return track_data
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            logger.error(f"Analysis failed for {file_path}: {e}")
            return None
    
    def _get_file_info(self, file_path: str) -> dict:
        """Get basic file information"""
        stat = os.stat(file_path)
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'modified_time': stat.st_mtime,
            'extension': Path(file_path).suffix.lower()
        }
    
    def _extract_metadata(self, file_path: str) -> dict:
        """Extract metadata using Mutagen"""
        try:
            file = mutagen.File(file_path)
            if not file:
                return {}
            
            # Extract common metadata fields
            metadata = {}
            
            # Handle different tag formats
            if hasattr(file, 'tags') and file.tags:
                tags = file.tags
                
                # Common mappings for different formats
                title_keys = ['TIT2', 'TITLE', '\xa9nam']
                artist_keys = ['TPE1', 'ARTIST', '\xa9ART'] 
                album_keys = ['TALB', 'ALBUM', '\xa9alb']
                year_keys = ['TDRC', 'DATE', '\xa9day']
                genre_keys = ['TCON', 'GENRE', '\xa9gen']
                
                metadata['title'] = self._get_first_value(tags, title_keys)
                metadata['artist'] = self._get_first_value(tags, artist_keys)
                metadata['album'] = self._get_first_value(tags, album_keys)
                metadata['year'] = self._get_first_value(tags, year_keys)
                metadata['genre'] = self._get_first_value(tags, genre_keys)
            
            # Audio properties
            if hasattr(file, 'info') and file.info:
                metadata['duration'] = getattr(file.info, 'length', 0)
                metadata['bitrate'] = getattr(file.info, 'bitrate', 0)
                metadata['sample_rate'] = getattr(file.info, 'sample_rate', 0)
                metadata['channels'] = getattr(file.info, 'channels', 0)
            
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
    
    async def _get_external_data(self, metadata: dict) -> dict:
        """Get external API data (Last.fm, MusicBrainz, etc.)"""
        external_data = {}
        
        artist = metadata.get('artist')
        title = metadata.get('title')
        
        if artist and title:
            # Try external APIs
            lastfm_data = await external_api_manager.get_lastfm_data(artist, title)
            if lastfm_data:
                external_data['lastfm'] = lastfm_data
            
            musicbrainz_data = await external_api_manager.get_musicbrainz_data(artist, title)
            if musicbrainz_data:
                external_data['musicbrainz'] = musicbrainz_data
            
            spotify_data = await external_api_manager.get_spotify_data(artist, title)
            if spotify_data:
                external_data['spotify'] = spotify_data
        
        return external_data
    
    async def _store_in_database(self, file_path: str, file_info: dict, metadata: dict, audio_features: dict, external_data: dict) -> dict:
        """Store all extracted data in PostgreSQL database"""
        
        # Generate unique ID
        track_id = str(uuid.uuid4())
        
        # Prepare track data
        track_data = {
            'id': track_id,
            'file_path': file_path,
            'filename': os.path.basename(file_path),
            'title': metadata.get('title'),
            'artist': metadata.get('artist'),
            'album': metadata.get('album'),
            'year': metadata.get('year'),
            'genre': metadata.get('genre'),
            'duration': metadata.get('duration', 0),
            'file_size': file_info['size_bytes'],
            'file_format': file_info['extension'],
            'bitrate': metadata.get('bitrate'),
            'sample_rate': metadata.get('sample_rate'),
            'channels': metadata.get('channels'),
            'audio_features': audio_features,
            'metadata': metadata,
            'external_data': external_data
        }
        
        # Store in database
        async with get_db_session() as session:
            track = Track(
                id=track_data['id'],
                file_path=track_data['file_path'],
                filename=track_data['filename'],
                title=track_data['title'],
                artist=track_data['artist'],
                album=track_data['album'],
                duration=track_data['duration'],
                file_size=track_data['file_size'],
                audio_features=track_data['audio_features']
            )
            
            session.add(track)
            await session.commit()
            
            logger.info(f"Track stored in database: {track_id}")
        
        return track_data

async def analyze_music_library():
    """Analyze music library and store in database"""
    
    print("🎵 COMPLETE ANALYSIS & DATABASE STORAGE")
    print("=" * 50)
    
    # Initialize
    await model_manager.load_models()
    workflow = DatabaseAnalysisWorkflow()
    
    # Find music files
    music_dir = "/music"
    audio_files = []
    
    print(f"📁 Scanning {music_dir}...")
    for root, dirs, files in os.walk(music_dir):
        for file in files:
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg')):
                audio_files.append(os.path.join(root, file))
    
    print(f"🎯 Found {len(audio_files)} audio files")
    
    # Analyze first 5 files (for demonstration)
    sample_files = audio_files[:5]
    
    print(f"\n🔬 Analyzing {len(sample_files)} files for demonstration...")
    print("=" * 50)
    
    results = []
    for i, file_path in enumerate(sample_files, 1):
        print(f"\n[{i}/{len(sample_files)}] Processing: {os.path.basename(file_path)}")
        
        result = await workflow.analyze_and_store_track(file_path)
        if result:
            results.append(result)
    
    # Show database content
    await show_database_content()
    
    print(f"\n🎯 Analysis Complete!")
    print(f"   ✅ Analyzed: {workflow.analyzed_count} tracks")
    print(f"   💾 Stored in database: {len(results)} records")
    print(f"   🗄️ Database: PostgreSQL playlista_v2")
    
    return results

async def show_database_content():
    """Show what's actually stored in the database"""
    
    print(f"\n📊 DATABASE CONTENT")
    print("=" * 50)
    
    try:
        async with get_db_session() as session:
            from sqlalchemy import text
            
            # Count tracks
            result = await session.execute(text("SELECT COUNT(*) FROM tracks"))
            track_count = result.scalar()
            
            print(f"📄 Total tracks in database: {track_count}")
            
            if track_count > 0:
                # Show sample tracks
                result = await session.execute(text("""
                    SELECT id, title, artist, album, duration, 
                           jsonb_extract_path_text(audio_features, 'tempo') as tempo,
                           jsonb_extract_path_text(audio_features, 'key') as key
                    FROM tracks 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """))
                
                tracks = result.fetchall()
                
                print(f"\n📋 Recent tracks:")
                for track in tracks:
                    title = track[1] or 'Unknown'
                    artist = track[2] or 'Unknown' 
                    tempo = track[5] or 'N/A'
                    key = track[6] or 'N/A'
                    print(f"   🎵 {title} by {artist} (Tempo: {tempo}, Key: {key})")
                
                # Show feature summary
                result = await session.execute(text("""
                    SELECT jsonb_object_keys(audio_features) as feature_key, COUNT(*) 
                    FROM tracks 
                    WHERE audio_features IS NOT NULL
                    GROUP BY feature_key
                    LIMIT 10
                """))
                
                features = result.fetchall()
                if features:
                    print(f"\n📊 Available audio features:")
                    for feature, count in features:
                        print(f"   🔬 {feature}: {count} tracks")
                        
            else:
                print("   ❌ No tracks found - run analysis first")
                
    except Exception as e:
        print(f"   ❌ Database query failed: {e}")

async def main():
    """Main function"""
    await analyze_music_library()

if __name__ == "__main__":
    asyncio.run(main())
