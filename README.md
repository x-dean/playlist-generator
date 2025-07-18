# Music Playlist Generator

An optimized music playlist generator that creates thematic playlists based on audio features. Uses machine learning clustering to group similar tracks with descriptive playlist names.

## Features

- **Audio Feature Analysis**: Extracts BPM, spectral centroid, duration, and beat confidence  
- **Intelligent Clustering**: Groups tracks using MiniBatch K-Means algorithm  
- **Descriptive Playlist Names**: Clear naming based on audio characteristics  
- **SQLite Database**: Caches analysis results for faster processing  
- **Automatic Updates**: Handles file additions/removals through metadata tracking  
- **Parallel Processing**: Utilizes multiple CPU cores for faster analysis  
- **Linux Compatibility**: Generates filesystem-friendly playlist names  

## Playlist Naming Convention

Playlists follow this naming pattern:  
`[Tempo]_[Timbre]_[Duration]_E[EnergyLevel]_[Mood]`

### Classification System

1. **Tempo (BPM)**:
   - `VerySlow`: <55 BPM  
   - `Slow`: 55–70 BPM  
   - `Chill`: 70–85 BPM  
   - `Medium`: 85–100 BPM  
   - `Upbeat`: 100–115 BPM  
   - `Energetic`: 115–135 BPM  
   - `Fast`: 135–155 BPM  
   - `VeryFast`: 155+ BPM  

2. **Timbre (Spectral Centroid)**:
   - `Dark`: <800 Hz  
   - `Warm`: 800–1500 Hz  
   - `Mellow`: 1500–2200 Hz  
   - `Bright`: 2200–3000 Hz  
   - `Sharp`: 3000–4000 Hz  
   - `VerySharp`: 4000+ Hz  

3. **Duration**:
   - `Brief`: <60 seconds  
   - `Short`: 60–120 seconds  
   - `Medium`: 120–180 seconds  
   - `Long`: 180–240 seconds  
   - `VeryLong`: 240+ seconds  

4. **Energy Level (0–10)**:
   - Scaled from beat confidence  
   - `E0` = low energy, `E10` = high energy  

5. **Mood**:
   - `Ambient`: Very slow with dark timbre  
   - `Downtempo`: Slow with low energy  
   - `Dance`: Fast with high energy  
   - `Dynamic`: Bright timbre with medium-high energy  
   - `Balanced`: Default for other combinations  

### Example Names

- `Medium_Bright_Medium_E6_Balanced.m3u`  
- `Chill_Mellow_Long_E3_Downtempo.m3u`  
- `Fast_Sharp_Brief_E9_Dance.m3u`  

## Database and Cache System

The system uses a SQLite database for efficient processing:

- **Automatic Caching**: Analysis results are stored in `/app/cache/audio_analysis.db`  
- **Metadata Tracking**: Uses file size and modification time to detect changes  
- **Smart Updates**:
  - Re-analyzes files when modified  
  - Skips unchanged files on subsequent runs  
  - Automatically handles file additions/removals  
- **Manual Refresh**: Delete cache files to force full re-analysis  

## Requirements

- Docker  
- Docker Compose  

## Installation

```bash
git clone https://github.com/x-dean/music-playlist-generator.git

cd music-playlist-generator
```
## Configure run.sh
```
# Edit these variables in run.sh
MUSIC_DIR="/path/to/your/music"
OUTPUT_DIR="${PWD}/playlists"
CACHE_DIR="${PWD}/cache"
```

# Run the generator
```
./run.sh \
  --workers=8 \
  --num_playlists=12
```

## Command Options

| Parameter            | Default        | Description                          |
|---------------------|----------------|--------------------------------------|
| `--music_dir`        | `/music`       | Music directory to analyze           |
| `--output_dir`       | `./playlists`  | Playlist output directory            |
| `--workers`          | CPU cores / 2  | Parallel processing threads          |
| `--num_playlists`    | 10             | Target number of playlists           |
| `--chunk_size`       | 1000           | Clustering batch size                |
| `--use_db`           | false          | Use cached database analysis         |
| `--force_sequential` | false          | Disable parallel processing          |
| `--rebuild`          | false          | Rebuild Docker image before running  |

## Docker Image

Pre-built image available on Docker Hub:

```bash
docker pull deanxaka/playlist-generator:latest
```

## Output

- **Generated Playlists**:  
  `.m3u` files will be created in the specified output directory.

- **Failed Files List**:  
  `Failed_Files.m3u` will contain a list of music files that could not be processed.

- **Log File**:  
  `playlist_generator.log` provides detailed information about the processing and any errors encountered.

## Customization

You can modify the analysis behavior and playlist generation logic by editing the following files:

- **`analyze_music.py`**:  
  - Configure timeouts  
  - Customize audio feature extraction logic

- **`playlist_generator.py`**:  
  - Tune playlist naming thresholds  
  - Adjust clustering parameters

### Example: Adjusting Playlist Naming

To change how playlists are named based on BPM, edit the `generate_playlist_name` method in `playlist_generator.py`:

```python
def generate_playlist_name(self, features):
    # Adjust BPM thresholds
    bpm = features['bpm']
    bpm_desc = (
        "Relaxed" if bpm < 60 else
        "Moderate" if bpm < 90 else
        "Active" if bpm < 120 else
        "Energetic" if bpm < 150 else
        "Intense"
    )
    # ... rest of the method
```

## Rebuilding the Docker Image

To force a complete rebuild of the Docker container (useful after making code changes):

```bash
./run.sh --rebuild
```

