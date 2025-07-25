# Playlist Generator: Command Reference & Feature Testing

This guide explains every supported command and flag, what it does, and how to test each feature independently. Use this to verify, debug, and improve each part of the playlist generator.

---

## 1. Modes of Operation (Mutually Exclusive)

| Command/Flag         | What it Does                                                                 | How to Test |
|----------------------|------------------------------------------------------------------------------|-------------|
| `-a`, `--analyze_only` | **Analyze Only:** Scans all audio files, extracts features, updates DB. No playlists created. | `./run.sh -a ...`<br>Check DB for new features, no playlists generated. |
| `-g`, `--generate_only` | **Generate Only:** Uses cached features in DB to generate playlists. No audio analysis. | `./run.sh -g ...`<br>Check output directory for new playlists. |
| `-u`, `--update`     | **Update:** Regenerates all playlists from DB and updates DB tables.         | `./run.sh -u ...`<br>Check that all playlists are refreshed. |
| *(none)*             | **Full Pipeline:** Analyzes audio and generates playlists in one run.        | `./run.sh ...`<br>Check both DB and output for new features/playlists. |

---

## 2. Playlist Generation Methods

| Command/Flag         | What it Does                                                                 | How to Test |
|----------------------|------------------------------------------------------------------------------|-------------|
| `-m all` (default)   | **Feature-grouping:** Robust, musically meaningful playlists.                | `./run.sh -g -m all ...`<br>Check output for playlists by musical features. |
| `-m time`            | **Time-based:** Playlists for each time slot (Morning, Afternoon, etc.).     | `./run.sh -g -m time ...`<br>Check output for time slot playlists. |
| `-m kmeans`          | **KMeans:** Clusters tracks by features, always assigns all tracks.          | `./run.sh -g -m kmeans ...`<br>Check output for cluster-based playlists. |
| `-m cache`           | **Cache/rule-based:** Groups tracks by feature bins (legacy/alternative).    | `./run.sh -g -m cache ...`<br>Check output for rule-based playlists. |
| `-m tags`            | **Tag-based:** Playlists by genre+decade, requires enough tracks per genre.  | `./run.sh -g -m tags ...`<br>Check output for genre/decade playlists. |

---

## 3. Status & Statistics

| Command/Flag         | What it Does                                                                 | How to Test |
|----------------------|------------------------------------------------------------------------------|-------------|
| `--status`           | Shows database/library statistics and exits.                                 | `./run.sh --status ...`<br>Check output for stats panel. |

---

## 4. Other Options

| Command/Flag         | What it Does                                                                 | How to Test |
|----------------------|------------------------------------------------------------------------------|-------------|
| `--workers N`        | Number of parallel workers for analysis.                                     | `./run.sh -a --workers 4 ...`<br>Check CPU usage and speed. |
| `--num_playlists N`  | Number of playlists to generate (for kmeans).                               | `./run.sh -g -m kmeans --num_playlists 10 ...`<br>Check number of playlists. |
| `--output_dir DIR`   | Where to save playlists.                                                     | `./run.sh ... --output_dir=/my/output ...`<br>Check output location. |
| `--music_dir DIR`    | Music directory (container path).                                            | `./run.sh ... --music_dir=/music ...` |
| `--library DIR` | Music directory (host path).                                               | `./run.sh ... --library=/music ...` |
| `--min_tracks_per_genre N` | Minimum tracks for a genre playlist (tags method).                    | `./run.sh -g -m tags --min_tracks_per_genre 15 ...`<br>Check which genres get playlists. |
| `--force_sequential` | Forces sequential processing (no parallelism).                              | `./run.sh -a --force_sequential ...`<br>Check for single-threaded operation. |

---

## 6. Help & Rebuild

| Command/Flag         | What it Does                                                                 | How to Test |
|----------------------|------------------------------------------------------------------------------|-------------|
| `--help`, `-h`       | Shows help message with all options.                                         | `./run.sh --help` |
| `--rebuild`          | Rebuilds the Docker image.                                                   | `./run.sh --rebuild ...` |

---

## How to Use This Table

- **Test each feature independently:** Run the command, observe the output, and check the expected result.
- **If a feature does not work as described:** Note the behavior and we can debug or improve it as a separate task.
- **Want to add or change a feature?** Just ask for a new row or a new section. 