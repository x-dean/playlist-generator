# 🎵 Playlist Generator

## Overview

This tool analyzes your music library, extracts audio features, and generates playlists grouped by musical characteristics. It supports multiple playlist generation methods (feature-group, time-based, kmeans, cache) and is designed for robust, radio-ready output.

---

## 1. Modes of Operation

You must choose **one** of the following modes for each run:

| Mode                | Short Flag | What it Does                                                                 |
|---------------------|:----------:|------------------------------------------------------------------------------|
| Analyze Only        |   `-a`     | Analyzes all audio files, updates the feature database. No playlists created. |
| Generate Only       |   `-g`     | Generates playlists from the database. No audio analysis is performed.        |
| Update              |   `-u`     | Regenerates **all** playlists from the database and updates the DB.           |
| Full Pipeline       |   (none)   | Analyzes audio **and** generates playlists in one run.                        |

> **Note:** You must specify one of `-a`, `-g`, or `-u`. If none is given, it defaults to `-a` (analyze only).

---

## 2. Playlist Generation Methods

Choose with `-m` or `--playlist_method`:

| Method         | Description                                                                                 |
|----------------|--------------------------------------------------------------------------------------------|
| all (default)  | Feature-grouping: robust, musically meaningful playlists (recommended for most users)      |
| time           | Playlists for each time slot (Morning, Afternoon, etc.), split if too long                 |
| kmeans         | Clusters tracks by audio features using k-means, always assigns all tracks                 |
| cache          | Rule-based grouping by feature bins (similar to “all”, fallback for legacy compatibility)  |

---

## 3. Example Commands

### A. Analyze Your Music Library
```sh
./run.sh -a --music_dir=/path/to/music --host_music_dir=/path/to/music --output_dir=/path/to/playlists
```
- Scans all audio files, extracts features, and updates the database.

### B. Generate Playlists (from DB, no re-analysis)
```sh
./run.sh -g --music_dir=/path/to/music --host_music_dir=/path/to/music --output_dir=/path/to/playlists
```
- Uses cached features to generate playlists (default: feature-group method).

### C. Update All Playlists (regenerate from DB)
```sh
./run.sh -u --music_dir=/path/to/music --host_music_dir=/path/to/music --output_dir=/path/to/playlists
```
- Regenerates all playlists from the database and updates the DB tables.

### D. Generate Time-Based Playlists
```sh
./run.sh -g -m time --music_dir=/path/to/music --host_music_dir=/path/to/music --output_dir=/path/to/playlists
```
- Creates playlists for each time slot (e.g., Morning, Afternoon), splitting if the total duration exceeds the slot.

### E. Generate KMeans Playlists
```sh
./run.sh -g -m kmeans --music_dir=/path/to/music --host_music_dir=/path/to/music --output_dir=/path/to/playlists
```
- Clusters tracks by features, always assigns all tracks, fallback to “Mixed_Collection” if needed.

### F. Generate Cache-Based Playlists
```sh
./run.sh -g -m cache --music_dir=/path/to/music --host_music_dir=/path/to/music --output_dir=/path/to/playlists
```
- Groups tracks by feature bins (legacy/alternative to “all”).

### G. Control Minimum Tracks per Genre for Tag-Based Playlists

When using the tag-based playlist method (`-m tags`), you can control the minimum number of tracks required for a genre to generate a playlist using the `--min_tracks_per_genre` option.

- **Purpose:** Avoids creating playlists for rare genres with only a few tracks.
- **Default:** 10 tracks per genre.
- **Usage Example:**

```sh
./run.sh -g -m tags --min_tracks_per_genre 15 --music_dir=/music --host_music_dir=/music --output_dir=/playlists
```

This will only create genre+decade playlists for genres that have at least 15 tracks in your library.

You can adjust the threshold as needed. Genres with fewer tracks will be ignored for playlist creation.

---

## 4. Output

- Playlists are saved as `.m3u` files in your output directory.
- Each playlist is named by its musical characteristics (e.g., `Upbeat_Groovy_C_Major_Bright.m3u`, `TimeSlot_Morning_Part1.m3u`).
- A `Failed_Files.m3u` is created for any files that could not be analyzed.

---

## 5. For Radio Automation (e.g., Liquidsoap)

- Use the generated `.m3u` files as playlist sources:
    ```liquidsoap
    morning = playlist("/path/to/playlists/TimeSlot_Morning.m3u")
    midday = playlist("/path/to/playlists/TimeSlot_Midday.m3u")
    radio = rotate([morning, midday, ...])
    ```
- If a time slot is split (e.g., `TimeSlot_Morning_Part1.m3u`, `TimeSlot_Morning_Part2.m3u`), you can rotate or shuffle these sub-playlists for variety.

---

## 6. Best Practices

- **Analyze first, then generate:**  
  Run `-a` to analyze, then `-g` or `-u` to generate playlists. This is fastest and avoids unnecessary re-analysis.
- **Update regularly:**  
  Use `-u` after adding new music to keep playlists fresh.
- **Choose the right method:**  
  Use `all` for most cases, `time` for dayparting, `kmeans` for experimental clustering, `cache` for legacy compatibility.
- **Check logs:**  
  The tool logs playlist sizes, fallback behavior, and any errors for easy troubleshooting.

---

## 7. Troubleshooting

- **No playlists generated?**  
  - Check that you used `-g` or `-u`.
  - Check logs for feature extraction issues.
  - Lower the minimum playlist size if your library is small.

- **Playlists too large?**  
  - The tool splits large playlists automatically (default max: 500 tracks).

- **Want more/fewer playlists?**  
  - Adjust `--num_playlists` (for kmeans) or tune feature rules.

---

## 8. Advanced

- **Customizing playlist logic:**  
  Edit `playlist_generator/playlist_generator/feature_group.py`, `kmeans.py`, `cache.py`, or `time_based.py` to change grouping, splitting, or naming logic.
- **Environment variables:**  
  You can set `MIN_PLAYLIST_SIZE` and `MAX_PLAYLIST_SIZE` to control playlist sizes.

---

## 9. Quick Reference Table

| Flag/Option         | Purpose                                      |
|---------------------|----------------------------------------------|
| `-a`                | Analyze only                                 |
| `-g`                | Generate playlists only                      |
| `-u`                | Update all playlists from DB                 |
| `-m all`            | Feature-group playlists (default)            |
| `-m time`           | Time-based playlists                         |
| `-m kmeans`         | KMeans clustering playlists                  |
| `-m cache`          | Cache/rule-based playlists                   |
| `--num_playlists N` | Number of playlists (for kmeans)             |
| `--workers N`       | Number of parallel workers                   |
| `--output_dir DIR`  | Where to save playlists                      |
| `--music_dir DIR`   | Music directory (container path)             |
| `--host_music_dir DIR` | Music directory (host path)               |
| `--min_tracks_per_genre N` | Minimum tracks required for a genre to create a playlist (tags method) |

---

## 10. Example: Full Workflow

```sh
# 1. Analyze your music library
./run.sh -a --music_dir=/music --host_music_dir=/music --output_dir=/playlists

# 2. Generate time-based playlists for radio
./run.sh -g -m time --music_dir=/music --host_music_dir=/music --output_dir=/playlists

# 3. Update all playlists after adding new music
./run.sh -u --music_dir=/music --host_music_dir=/music --output_dir=/playlists
```

---

If you need more customization, want to automate the workflow, or have questions about integration with your radio stack, just ask! 