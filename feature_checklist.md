# Playlist Generator Feature Testing Checklist

Use this checklist to test each feature independently. Mark each as complete (with a tick) and add comments/results as you go.

---

- [ ] **Analyze Only mode (`-a, --analyze_only`)**  
  **Command:** `./run.sh -a ...`  
  **Check:** DB updated, no playlists created.  
  **Comments:**

- [ ] **Generate Only mode (`-g, --generate_only`)**  
  **Command:** `./run.sh -g ...`  
  **Check:** Playlists generated from DB, no analysis.  
  **Comments:**

- [ ] **Update mode (`-u, --update`)**  
  **Command:** `./run.sh -u ...`  
  **Check:** All playlists regenerated from DB.  
  **Comments:**

- [ ] **Full Pipeline (no mode flag)**  
  **Command:** `./run.sh ...`  
  **Check:** Both analysis and playlist generation occur.  
  **Comments:**

- [ ] **Feature-grouping playlist method (`-m all`)**  
  **Command:** `./run.sh -g -m all ...`  
  **Check:** Playlists by musical features.  
  **Comments:**

- [ ] **Time-based playlist method (`-m time`)**  
  **Command:** `./run.sh -g -m time ...`  
  **Check:** Time slot playlists.  
  **Comments:**

- [ ] **KMeans playlist method (`-m kmeans`)**  
  **Command:** `./run.sh -g -m kmeans ...`  
  **Check:** Cluster-based playlists.  
  **Comments:**

- [ ] **Cache/rule-based playlist method (`-m cache`)**  
  **Command:** `./run.sh -g -m cache ...`  
  **Check:** Rule-based playlists.  
  **Comments:**

- [ ] **Tag-based playlist method (`-m tags`)**  
  **Command:** `./run.sh -g -m tags ...`  
  **Check:** Genre/decade playlists.  
  **Comments:**

- [ ] **Enrich missing tags (`--enrich_tags`)**  
  **Command:** `./run.sh --enrich_tags ...`  
  **Check:** DB metadata updated.  
  **Comments:**

- [ ] **Force enrich all tags (`--enrich_tags --force_enrich_tags`)**  
  **Command:** `./run.sh --enrich_tags --force_enrich_tags ...`  
  **Check:** All tags refreshed.  
  **Comments:**

- [ ] **Enrich only (`--enrich_only`)**  
  **Command:** `./run.sh --enrich_only ...`  
  **Check:** Only DB metadata updated, no playlists generated.  
  **Comments:**

- [ ] **Force with enrich only (`--enrich_only --force`)**  
  **Command:** `./run.sh --enrich_only --force ...`  
  **Check:** All tags refreshed.  
  **Comments:**

- [ ] **Status & Statistics (`--status`)**  
  **Command:** `./run.sh --status ...`  
  **Check:** Stats panel is printed.  
  **Comments:**

- [ ] **--workers N**  
  **Command:** `./run.sh -a --workers 4 ...`  
  **Check:** CPU usage and speed.  
  **Comments:**

- [ ] **--num_playlists N**  
  **Command:** `./run.sh -g -m kmeans --num_playlists 10 ...`  
  **Check:** Number of playlists.  
  **Comments:**

- [ ] **--output_dir DIR**  
  **Command:** `./run.sh ... --output_dir=/my/output ...`  
  **Check:** Output location.  
  **Comments:**

- [ ] **--min_tracks_per_genre N**  
  **Command:** `./run.sh -g -m tags --min_tracks_per_genre 15 ...`  
  **Check:** Which genres get playlists.  
  **Comments:**

- [ ] **--force_sequential**  
  **Command:** `./run.sh -a --force_sequential ...`  
  **Check:** Single-threaded operation.  
  **Comments:**

- [ ] **--help**  
  **Command:** `./run.sh --help`  
  **Check:** Help message is printed.  
  **Comments:**

- [ ] **--rebuild**  
  **Command:** `./run.sh --rebuild ...`  
  **Check:** Docker image is rebuilt.  
  **Comments:** 