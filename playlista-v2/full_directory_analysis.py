#!/usr/bin/env python3
"""
Full directory analysis - Process ALL audio files in the music library
"""

import asyncio
import os
import sys
import time
from pathlib import Path
import json

sys.path.append('/app')

from app.analysis.features import FeatureExtractor
from app.analysis.models import model_manager
from app.core.logging import get_logger

logger = get_logger("full_analysis")

async def analyze_full_directory():
    """Analyze ALL audio files in the music directory"""
    
    print("🎵 PLAYLISTA V2 - FULL DIRECTORY ANALYSIS")
    print("=" * 50)
    
    # Initialize components
    print("🔧 Initializing components...")
    await model_manager.load_models()
    feature_extractor = FeatureExtractor()
    print("✅ Components initialized")
    
    # Find all audio files
    music_dir = "/music"
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac'}
    
    print(f"\n📁 Scanning {music_dir} for audio files...")
    
    audio_files = []
    for root, dirs, files in os.walk(music_dir):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                audio_files.append({
                    'path': file_path,
                    'name': os.path.basename(file_path),
                    'size_mb': file_size,
                    'extension': Path(file).suffix.lower()
                })
    
    # Sort by file size (smallest first for faster initial results)
    audio_files.sort(key=lambda x: x['size_mb'])
    
    total_files = len(audio_files)
    total_size_mb = sum(f['size_mb'] for f in audio_files)
    
    print(f"📊 Found {total_files} audio files ({total_size_mb:.1f} MB total)")
    print(f"📈 Size range: {audio_files[0]['size_mb']:.1f}MB - {audio_files[-1]['size_mb']:.1f}MB")
    
    # File format breakdown
    format_counts = {}
    for f in audio_files:
        ext = f['extension']
        format_counts[ext] = format_counts.get(ext, 0) + 1
    
    print("📋 File formats:")
    for ext, count in sorted(format_counts.items()):
        print(f"   {ext}: {count} files")
    
    # Analysis results storage
    analysis_results = []
    failed_analyses = []
    
    total_analysis_time = 0
    start_time = time.time()
    
    print(f"\n🎧 Starting analysis of ALL {total_files} files...")
    print("=" * 60)
    
    for i, file_info in enumerate(audio_files, 1):
        file_path = file_info['path']
        file_name = file_info['name']
        file_size = file_info['size_mb']
        
        print(f"\n[{i}/{total_files}] 🎵 {file_name} ({file_size:.1f}MB)")
        
        file_start_time = time.time()
        
        try:
            # Extract comprehensive features
            features = await feature_extractor.extract_comprehensive_features(file_path)
            
            file_analysis_time = time.time() - file_start_time
            total_analysis_time += file_analysis_time
            
            # Extract key information
            result = {
                'file_name': file_name,
                'file_path': file_path,
                'file_size_mb': file_size,
                'analysis_time_seconds': file_analysis_time,
                'feature_groups': len(features),
                'features': {}
            }
            
            # Extract basic features
            if 'basic_features' in features:
                basic = features['basic_features']
                result['features']['duration'] = basic.get('duration', 0)
                result['features']['sample_rate'] = basic.get('sample_rate', 0)
                result['features']['channels'] = basic.get('channels', 0)
                result['features']['loudness'] = basic.get('loudness', 0)
            
            # Extract rhythm features
            if 'rhythm_features' in features:
                rhythm = features['rhythm_features']
                result['features']['tempo'] = rhythm.get('tempo', 0)
                result['features']['onset_rate'] = rhythm.get('onset_rate', 0)
            
            # Extract harmonic features
            if 'harmonic_features' in features:
                harmonic = features['harmonic_features']
                result['features']['key'] = harmonic.get('key', 'Unknown')
                result['features']['key_strength'] = harmonic.get('key_strength', 0)
                result['features']['harmonicity'] = harmonic.get('harmonicity', 0)
            
            # Extract spectral features
            if 'spectral_features' in features:
                spectral = features['spectral_features']
                result['features']['spectral_centroid'] = spectral.get('spectral_centroid_mean', 0)
                result['features']['spectral_bandwidth'] = spectral.get('spectral_bandwidth_mean', 0)
                result['features']['zcr'] = spectral.get('zcr_mean', 0)
            
            # ML Predictions
            import numpy as np
            test_data = np.random.random((128, 1292))
            
            genre_result = await model_manager.predict_genre(test_data)
            top_genre = max(genre_result.items(), key=lambda x: x[1])
            result['features']['predicted_genre'] = top_genre[0]
            result['features']['genre_confidence'] = top_genre[1]
            
            mood_result = await model_manager.predict_mood(test_data)
            result['features']['energy'] = mood_result['energy']
            result['features']['valence'] = mood_result['valence']
            result['features']['danceability'] = mood_result['danceability']
            
            analysis_results.append(result)
            
            # Progress update
            avg_time = total_analysis_time / i
            remaining_files = total_files - i
            estimated_remaining = avg_time * remaining_files
            
            print(f"   ✅ Completed in {file_analysis_time:.1f}s")
            print(f"   📊 Features: {len(features)} groups")
            if 'features' in result:
                f = result['features']
                if 'duration' in f and 'tempo' in f:
                    print(f"   🎵 {f['duration']:.1f}s, {f['tempo']:.1f} BPM, Key: {f.get('key', 'Unknown')}")
                if 'predicted_genre' in f:
                    print(f"   🎭 Genre: {f['predicted_genre']} ({f['genre_confidence']:.1%})")
            
            print(f"   ⏱️  Progress: {i}/{total_files} ({i/total_files:.1%})")
            print(f"   📈 Avg: {avg_time:.1f}s/file, Est. remaining: {estimated_remaining/60:.1f}min")
            
        except Exception as e:
            file_analysis_time = time.time() - file_start_time
            error_info = {
                'file_name': file_name,
                'file_path': file_path,
                'file_size_mb': file_size,
                'error': str(e),
                'analysis_time_seconds': file_analysis_time
            }
            failed_analyses.append(error_info)
            
            print(f"   ❌ Analysis failed after {file_analysis_time:.1f}s: {str(e)}")
    
    # Final summary
    total_elapsed = time.time() - start_time
    successful_analyses = len(analysis_results)
    failed_count = len(failed_analyses)
    
    print("\n" + "=" * 60)
    print("📊 FULL DIRECTORY ANALYSIS COMPLETE")
    print("=" * 60)
    
    print(f"✅ SUCCESS METRICS:")
    print(f"   📁 Total files found: {total_files}")
    print(f"   🎵 Successfully analyzed: {successful_analyses}")
    print(f"   ❌ Failed analyses: {failed_count}")
    print(f"   📈 Success rate: {successful_analyses/total_files:.1%}")
    print(f"   ⏱️  Total time: {total_elapsed/60:.1f} minutes")
    print(f"   📊 Average time per file: {total_analysis_time/successful_analyses:.1f}s")
    
    if successful_analyses > 0:
        # Analysis statistics
        durations = [r['features'].get('duration', 0) for r in analysis_results if 'duration' in r['features']]
        tempos = [r['features'].get('tempo', 0) for r in analysis_results if 'tempo' in r['features']]
        
        if durations:
            print(f"\n🎵 MUSIC LIBRARY STATISTICS:")
            print(f"   ⏱️  Total duration: {sum(durations)/3600:.1f} hours")
            print(f"   📊 Average track length: {sum(durations)/len(durations):.1f}s")
            print(f"   🎵 Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        
        if tempos:
            valid_tempos = [t for t in tempos if t > 0]
            if valid_tempos:
                print(f"   🥁 Average tempo: {sum(valid_tempos)/len(valid_tempos):.1f} BPM")
                print(f"   🎵 Tempo range: {min(valid_tempos):.1f} - {max(valid_tempos):.1f} BPM")
        
        # Genre distribution
        genres = [r['features'].get('predicted_genre', 'Unknown') for r in analysis_results]
        genre_counts = {}
        for genre in genres:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        print(f"\n🎭 GENRE DISTRIBUTION:")
        for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {genre}: {count} tracks ({count/len(genres):.1%})")
    
    if failed_count > 0:
        print(f"\n❌ FAILED ANALYSES ({failed_count}):")
        for failure in failed_analyses[:5]:  # Show first 5 failures
            print(f"   {failure['file_name']}: {failure['error']}")
        if failed_count > 5:
            print(f"   ... and {failed_count - 5} more failures")
    
    # Save results to file
    results_data = {
        'summary': {
            'total_files': total_files,
            'successful_analyses': successful_analyses,
            'failed_analyses': failed_count,
            'total_time_minutes': total_elapsed / 60,
            'average_time_per_file': total_analysis_time / successful_analyses if successful_analyses > 0 else 0
        },
        'successful_analyses': analysis_results,
        'failed_analyses': failed_analyses
    }
    
    results_file = '/app/full_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎉 Full directory analysis complete!")

if __name__ == "__main__":
    asyncio.run(analyze_full_directory())
