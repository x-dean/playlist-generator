# 🗄️ Database Structure - Complete Data Flow

## 📊 Database Schema Overview

The PostgreSQL database captures **ALL analysis data** in a structured, web-optimized format:

### **🎵 Core Music Data**

#### **1. `tracks` Table** - Fast Query Optimized
```sql
-- File & Basic Info
file_path, file_hash, filename, file_size_bytes
title, artist, album, genre[], year, duration_seconds
bitrate, sample_rate, channels

-- Essential Features (indexed for fast playlist queries)
tempo, key, mode, key_confidence
energy, danceability, valence, acousticness
instrumentalness, liveness, speechiness, loudness

-- Analysis Tracking
analysis_completed, analysis_date, analysis_method
```

#### **2. `track_analysis` Table** - Complete Analysis Data
```sql
-- Essentia Features (JSON format preserves all details)
essentia_rhythm: {
  "tempo_mean": 128.5, "tempo_std": 0.2, "tempo_median": 128.3,
  "tempo_confidence_mean": 3.4, "beats_intervals": [...],
  "segments_analyzed": 4
}

essentia_spectral: {
  "spectral_centroid_mean": 2150.4, "spectral_centroid_std": 156.8,
  "loudness_mean": 2265.3, "loudness_std": 69.7
}

essentia_harmonic: {
  "key": "C", "scale": "major", "key_strength": 0.89,
  "key_confidence": 1.0, "chroma_features": [...]
}

essentia_mfcc: {
  "mfcc_coefficients": [...], "timbre_features": [...]
}

-- MusiCNN AI Results
musicnn_tags: {
  "electronic": 0.89, "dance": 0.67, "energetic": 0.75,
  "pop": 0.95, "happy": 0.29, "acoustic": 0.61, ...
}

musicnn_embeddings: [50-dimensional vector for similarity]
musicnn_confidence: 0.85

-- Processing Metadata
segments_analyzed: 4
segment_times: [[0, 30], [60, 90], [120, 150], [240, 270]]
processing_time_seconds: 45.2
cache_key: "032e4a719c6be5a3d849e84dd084ee7f"
```

#### **3. `music_tags` + `track_tags`** - Normalized Tag System
```sql
-- Normalized tag categories
music_tags: {genre, mood, instrument, style, vocal}
track_tags: {track_id, tag_id, confidence, source}

-- Example linkage:
Track "Amazing Song" -> 
  - "electronic" (confidence: 0.89, source: 'musicnn')
  - "dance" (confidence: 0.67, source: 'musicnn')
  - "energetic" (confidence: 0.75, source: 'musicnn')
```

---

## 🔄 Data Flow: From Audio File to Database

### **📁 Input: Audio File**
```
/music/Electronic/Artist - Song.mp3 (15.2 MB)
```

### **🔍 Step 1: File Discovery**
```sql
-- File identification
file_path: "/music/Electronic/Artist - Song.mp3"
file_hash: "abc123def456..." (content-based)
filename: "Artist - Song.mp3"
file_size_bytes: 15728640
```

### **🎵 Step 2: Metadata Extraction (Mutagen)**
```sql
-- ID3 tags → tracks table
title: "Amazing Song"
artist: "Great Artist"
album: "Epic Album"
genre: ["Electronic", "Dance"]
year: 2024
duration_seconds: 245.3
bitrate: 320000
sample_rate: 44100
channels: 2
```

### **🧠 Step 3: Audio Analysis**

#### **OptimizedPipeline Processing:**
```python
# Multi-segment analysis (5-200MB files)
1. Extract 4 segments (30s each) from different parts
2. Run Essentia on each segment
3. Run MusiCNN on each segment  
4. Aggregate results statistically

# Standard analysis (<5MB or >200MB files)
1. Analyze full track
2. Single Essentia + MusiCNN pass
```

#### **Essentia Features Extracted:**
```json
{
  "rhythm": {
    "tempo": 128.5,
    "tempo_confidence": 3.4,
    "beats_per_minute": 128.5,
    "beat_intervals": [0.468, 0.937, 1.406, ...]
  },
  "spectral": {
    "spectral_centroid_mean": 2150.4,
    "spectral_centroid_std": 156.8,
    "loudness": 2265.3,
    "spectral_rolloff": 4205.6
  },
  "harmonic": {
    "key": "C",
    "scale": "major", 
    "key_strength": 0.89,
    "chroma_vector": [0.8, 0.1, 0.6, ...]
  },
  "mfcc": {
    "coefficients": [-12.5, 8.3, -4.1, 2.7, ...]
  }
}
```

#### **MusiCNN AI Predictions:**
```json
{
  "tags": {
    "electronic": 0.89,
    "dance": 0.67,
    "pop": 0.95,
    "energetic": 0.75,
    "happy": 0.29,
    "acoustic": 0.61,
    "instrumental": 0.71,
    // ... 42 total categories
  },
  "embeddings": [-1.08, 0.99, 0.28, -1.50, ...], // 50 dimensions
  "confidence": 0.85
}
```

### **📊 Step 4: Feature Derivation**
```python
# Derive playlist features from MusiCNN tags
energy = f(tags["energetic"], tags["aggressive"], tags["dance"] - tags["peaceful"])
danceability = f(tags["dance"], tags["electronic"] - tags["classical"])
valence = f(tags["happy"] - tags["sad"], tags["uplifting"])
acousticness = f(tags["acoustic"], tags["folk"] - tags["electronic"])
```

### **💾 Step 5: Database Storage**

#### **Main Track Record (`tracks` table):**
```sql
INSERT INTO tracks (
  file_path, file_hash, filename, file_size_bytes,
  title, artist, album, genre, year, duration_seconds,
  tempo, key, mode, key_confidence,
  energy, danceability, valence, acousticness,
  analysis_completed, analysis_method
) VALUES (
  '/music/Electronic/Artist - Song.mp3',
  'abc123def456...',
  'Artist - Song.mp3', 
  15728640,
  'Amazing Song', 'Great Artist', 'Epic Album', 
  '["Electronic", "Dance"]', 2024, 245.3,
  128.5, 'C', 'major', 0.89,
  0.82, 0.75, 0.68, 0.31,
  true, 'optimized'
);
```

#### **Complete Analysis Data (`track_analysis` table):**
```sql
INSERT INTO track_analysis (
  track_id, 
  essentia_rhythm, essentia_spectral, essentia_harmonic, essentia_mfcc,
  musicnn_tags, musicnn_embeddings, musicnn_confidence,
  segments_analyzed, segment_times, processing_time_seconds,
  cache_key
) VALUES (
  123,
  '{"tempo": 128.5, "tempo_confidence": 3.4, ...}',
  '{"spectral_centroid_mean": 2150.4, ...}',
  '{"key": "C", "scale": "major", ...}',
  '{"coefficients": [-12.5, 8.3, ...]}',
  '{"electronic": 0.89, "dance": 0.67, ...}',
  '[-1.08, 0.99, 0.28, -1.50, ...]', 
  0.85,
  4, '[[0,30], [60,90], [120,150], [240,270]]', 45.2,
  'abc123def456'
);
```

#### **Normalized Tags (`music_tags` + `track_tags`):**
```sql
-- Link track to tags with confidence scores
INSERT INTO track_tags (track_id, tag_id, confidence, source) VALUES
(123, (SELECT id FROM music_tags WHERE name='electronic'), 0.89, 'musicnn'),
(123, (SELECT id FROM music_tags WHERE name='dance'), 0.67, 'musicnn'),
(123, (SELECT id FROM music_tags WHERE name='energetic'), 0.75, 'musicnn'),
-- ... for all 42 MusiCNN categories above confidence threshold
```

---

## 🔗 Data Relationships & Connections

### **🎵 Track-Centric Design**
```
Audio File → tracks (main record)
         ↓
         track_analysis (complete data)
         ↓
         track_tags → music_tags (normalized categories)
```

### **🎧 Playlist System**
```
users → playlists → playlist_tracks → tracks
     ↓           ↓                  ↓
preferences  generation_method   position
```

### **🔍 Similarity & Discovery**
```
tracks → track_similarities ← tracks
     ↓                         ↑
musicnn_embeddings ——vector similarity
```

---

## 📈 Query Examples

### **Fast Playlist Generation**
```sql
-- Electronic dance tracks 120-140 BPM, high energy
SELECT title, artist, tempo, energy, danceability 
FROM tracks 
WHERE tempo BETWEEN 120 AND 140 
  AND energy > 0.7 
  AND genre::text ILIKE '%electronic%'
ORDER BY danceability DESC 
LIMIT 20;
```

### **Similarity Search**
```sql
-- Find tracks similar to track #123
SELECT t.title, t.artist,
       ta.musicnn_embeddings <-> ref_ta.musicnn_embeddings as similarity
FROM tracks t
JOIN track_analysis ta ON t.id = ta.track_id
CROSS JOIN track_analysis ref_ta
WHERE ref_ta.track_id = 123 
  AND t.id != 123
ORDER BY similarity
LIMIT 10;
```

### **Tag-Based Discovery**
```sql
-- Find all electronic tracks with high confidence
SELECT t.title, t.artist, tt.confidence
FROM tracks t
JOIN track_tags tt ON t.id = tt.track_id
JOIN music_tags mt ON tt.tag_id = mt.id
WHERE mt.name = 'electronic' 
  AND tt.confidence > 0.8
ORDER BY tt.confidence DESC;
```

---

## ✅ What's Captured vs Your JSON Cache

### **✅ Everything From JSON Cache PLUS:**
- **Structured relationships** - No more hash-only files
- **Fast indexed queries** - Tempo, key, energy lookups
- **Vector similarity** - AI-powered recommendations  
- **Normalized tags** - Consistent genre/mood categorization
- **User & playlist management** - Web UI ready
- **Processing metadata** - Segments, timing, cache linkage

### **🎯 Benefits Over JSON Files:**
- **Queryable** - SQL instead of file scanning
- **Relational** - Connect tracks, playlists, users
- **Performant** - Indexes for fast lookups
- **Scalable** - Concurrent access, transactions
- **Web Ready** - Built for applications, not just analysis

**🎵 Your complete music analysis is now properly structured, queryable, and ready for sophisticated playlist generation and music discovery!**
