# Playlist Generator Feature Testing Checklist

Use this checklist to test each feature independently. Mark each as complete (with a tick) and add comments/results as you go.

---

- [ ] **Analyze Only mode (`-a, --analyze_only`)**  
  **Test Environment/Setup:** Start with a music directory containing audio files. The database can be empty or partially filled. No playlists should be present in the output directory.  
  **Command:** `./run.sh -a ...`  
  **Check:** DB updated, no playlists created.  
  **Expected Output:** No new playlists in output dir; database file (`audio_analysis.db`) updated with new/updated track features. Console shows analysis progress.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features;'`  
  **Comments:**

- [ ] **Generate Only mode (`-g, --generate_only`)**  
  **Test Environment/Setup:** The database should already contain analyzed tracks (run analyze first if needed). Output directory can be empty or contain old playlists.  
  **Command:** `./run.sh -g ...`  
  **Check:** Playlists generated from DB, no analysis.  
  **Expected Output:** New `.m3u` playlist files in output dir; no analysis progress shown in console.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM playlists;'`  
  **Comments:**

- [ ] **Update mode (`-u, --update`)**  
  **Test Environment/Setup:** The database should contain analyzed tracks and existing playlists. Output directory should have old playlists.  
  **Command:** `./run.sh -u ...`  
  **Check:** All playlists regenerated from DB.  
  **Expected Output:** All playlist files in output dir are refreshed (check timestamps); console shows playlist generation progress.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT name, last_updated FROM playlists ORDER BY last_updated DESC LIMIT 5;'`  
  **Comments:**

- [ ] **Full Pipeline (no mode flag)**  
  **Test Environment/Setup:** Music directory with audio files. DB and output dir can be empty or partially filled.  
  **Command:** `./run.sh ...`  
  **Check:** Both analysis and playlist generation occur.  
  **Expected Output:** Console shows both analysis and playlist generation progress; output dir and DB both updated.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features;'` and `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM playlists;'`  
  **Comments:**

- [ ] **Feature-grouping playlist method (`-m all`)**  
  **Test Environment/Setup:** DB should contain analyzed tracks. Output dir can be empty.  
  **Command:** `./run.sh -g -m all ...`  
  **Check:** Playlists by musical features.  
  **Expected Output:** Output dir contains playlists named by musical features (e.g., `Upbeat_Groovy_C_Major_Bright.m3u`).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT name FROM playlists;'`  
  **Comments:**

- [ ] **Time-based playlist method (`-m time`)**  
  **Test Environment/Setup:** DB should contain analyzed tracks. Output dir can be empty.  
  **Command:** `./run.sh -g -m time ...`  
  **Check:** Time slot playlists.  
  **Expected Output:** Output dir contains playlists for each time slot (e.g., `TimeSlot_Morning.m3u`).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db "SELECT name FROM playlists WHERE name LIKE 'TimeSlot_%';"`  
  **Comments:**

- [ ] **KMeans playlist method (`-m kmeans`)**  
  **Test Environment/Setup:** DB should contain analyzed tracks. Output dir can be empty.  
  **Command:** `./run.sh -g -m kmeans ...`  
  **Check:** Cluster-based playlists.  
  **Expected Output:** Output dir contains playlists named by cluster/feature (e.g., `Fast_Energetic_Bright.m3u`).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT name FROM playlists WHERE name LIKE "%Cluster%";'`  
  **Comments:**

- [ ] **Cache/rule-based playlist method (`-m cache`)**  
  **Test Environment/Setup:** DB should contain analyzed tracks. Output dir can be empty.  
  **Command:** `./run.sh -g -m cache ...`  
  **Check:** Rule-based playlists.  
  **Expected Output:** Output dir contains playlists grouped by feature bins (e.g., `Medium_Balanced_Balanced.m3u`).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT name FROM playlists WHERE name LIKE "%Balanced%";'`  
  **Comments:**

- [ ] **Tag-based playlist method (`-m tags`)**  
  **Test Environment/Setup:** DB should contain analyzed tracks with genre/year metadata. Output dir can be empty.  
  **Command:** `./run.sh -g -m tags ...`  
  **Check:** Genre/decade playlists.  
  **Expected Output:** Output dir contains playlists named by genre and decade (e.g., `Rock_1990s.m3u`).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT name FROM playlists WHERE name LIKE "%1990s%";'`  
  **Comments:**

- [ ] **Enrich missing tags (`--enrich_tags`)**  
  **Test Environment/Setup:** DB should contain tracks missing genre/year metadata.  
  **Command:** `./run.sh --enrich_tags ...`  
  **Check:** DB metadata updated.  
  **Expected Output:** Console shows enrichment progress; DB metadata for tracks missing genre/year is updated.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features WHERE json_extract(metadata, "$.genre") IS NOT NULL;'`  
  **Comments:**

- [ ] **Force enrich all tags (`--enrich_tags --force_enrich_tags`)**  
  **Test Environment/Setup:** DB with tracks (regardless of current metadata).  
  **Command:** `./run.sh --enrich_tags --force_enrich_tags ...`  
  **Check:** All tags refreshed.  
  **Expected Output:** Console shows enrichment progress; all tracks' metadata is refreshed in DB.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features WHERE json_extract(metadata, "$.genre") IS NOT NULL;'`  
  **Comments:**

- [ ] **Enrich only (`--enrich_only`)**  
  **Test Environment/Setup:** DB with tracks, some missing metadata.  
  **Command:** `./run.sh --enrich_only ...`  
  **Check:** Only DB metadata updated, no playlists generated.  
  **Expected Output:** Console shows enrichment progress; no new playlists in output dir.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features WHERE json_extract(metadata, "$.genre") IS NOT NULL;'`  
  **Comments:**

- [ ] **Force with enrich only (`--enrich_only --force`)**  
  **Test Environment/Setup:** DB with tracks (regardless of current metadata).  
  **Command:** `./run.sh --enrich_only --force ...`  
  **Check:** All tags refreshed.  
  **Expected Output:** Console shows enrichment progress; all tracks' metadata is refreshed in DB.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features WHERE json_extract(metadata, "$.genre") IS NOT NULL;'`  
  **Comments:**

- [ ] **Status & Statistics (`--status`)**  
  **Test Environment/Setup:** DB can be empty or filled.  
  **Command:** `./run.sh --status ...`  
  **Check:** Stats panel is printed.  
  **Expected Output:** Console prints a stats panel with total tracks, playlists, tags, etc.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features;'` and `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM playlists;'`  
  **Comments:**

- [ ] **--workers N**  
  **Test Environment/Setup:** Music directory with enough files to benefit from parallelism.  
  **Command:** `./run.sh -a --workers 4 ...`  
  **Check:** CPU usage and speed.  
  **Expected Output:** Analysis runs with specified number of workers (check system monitor or logs for parallelism).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features;'`  
  **Comments:**

- [ ] **--num_playlists N**  
  **Test Environment/Setup:** DB with enough tracks for multiple playlists.  
  **Command:** `./run.sh -g -m kmeans --num_playlists 10 ...`  
  **Check:** Number of playlists.  
  **Expected Output:** Output dir contains approximately N playlists (for kmeans method).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM playlists;'`  
  **Comments:**

- [ ] **--output_dir DIR**  
  **Test Environment/Setup:** Any valid DB and music directory.  
  **Command:** `./run.sh ... --output_dir=/my/output ...`  
  **Check:** Output location.  
  **Expected Output:** Playlists are saved in the specified output directory.  
  **DB Inspection Command:** (Check output directory, not DB)  
  **Comments:**

- [ ] **--min_tracks_per_genre N**  
  **Test Environment/Setup:** DB with tracks of various genres.  
  **Command:** `./run.sh -g -m tags --min_tracks_per_genre 15 ...`  
  **Check:** Which genres get playlists.  
  **Expected Output:** Only genres with at least N tracks get playlists in output dir.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT name FROM playlists;'`  
  **Comments:**

- [ ] **--force_sequential**  
  **Test Environment/Setup:** Music directory with enough files to notice difference between sequential and parallel.  
  **Command:** `./run.sh -a --force_sequential ...`  
  **Check:** Single-threaded operation.  
  **Expected Output:** Analysis runs in single-threaded mode (check logs or system monitor).  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features;'`  
  **Comments:**

- [ ] **--help**  
  **Test Environment/Setup:** Any environment.  
  **Command:** `./run.sh --help`  
  **Check:** Help message is printed.  
  **Expected Output:** Console prints help/usage information for all options.  
  **DB Inspection Command:** (Not applicable)  
  **Comments:**

- [ ] **--rebuild**  
  **Test Environment/Setup:** Any environment.  
  **Command:** `./run.sh --rebuild ...`  
  **Check:** Docker image is rebuilt.  
  **Expected Output:** Console shows Docker build output; image is rebuilt.  
  **DB Inspection Command:** (Not applicable)  
  **Comments:** 

- [ ] **--worker_max_mem_mb**  
  **Test Environment/Setup:** Music directory with large files or files that could cause high memory usage.  
  **Command:** `./run.sh -a --worker_max_mem_mb=512 ...`  
  **Check:** Worker processes should not exceed 512MB RAM; files that would cause excess memory usage are skipped with a warning.  
  **Expected Output:** Console shows warnings for files skipped due to memory; analysis completes without OOM.  
  **DB Inspection Command:** `sqlite3 /app/cache/audio_analysis.db 'SELECT COUNT(*) FROM audio_features;'`  
  **Comments:** 