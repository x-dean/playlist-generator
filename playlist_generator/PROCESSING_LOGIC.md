# Processing Logic - File Size Based Strategy

## 🎯 **New Processing Logic**

### **Rule: If file is over 200MB and not in failed then use sequential, then parallel**

## 📊 **How It Works**

### **1. File Classification**
```python
# Files are classified by size from database
BIG_FILE_SIZE_MB = 200  # 200MB threshold

# Get file sizes from database (stored during file discovery)
file_sizes = audio_db.get_file_sizes_from_db(file_paths_only)

for file_path in files_to_analyze:
    file_size_bytes = file_sizes.get(file_path, 0)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    if file_size_mb > BIG_FILE_SIZE_MB:
        big_files.append(file_path)      # > 200MB
    else:
        normal_files.append(file_path)   # ≤ 200MB
```

### **2. Processing Strategy**

#### **Step 1: Sequential Processing (Big Files)**
```python
if big_files:
    logger.info(f"Processing {len(big_files)} big files sequentially...")
    sequential_processor = SequentialProcessor()
    workers = 1  # Sequential processing uses 1 worker
    
    for features, filepath, db_write_success in sequential_processor.process(big_files, workers, force_reextract=force_reextract):
        all_results.append((features, filepath, db_write_success))
```

#### **Step 2: Parallel Processing (Normal Files)**
```python
if normal_files:
    logger.info(f"Processing {len(normal_files)} normal files in parallel...")
    parallel_processor = ParallelProcessor()
    workers = args.workers or max(1, mp.cpu_count())
    
    for features, filepath, db_write_success in parallel_processor.process(normal_files, workers, force_reextract=force_reextract):
        all_results.append((features, filepath, db_write_success))
```

## 🔄 **Processing Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File List     │    │   Size Check    │    │   Processing    │
│   (To Analyze)  │───▶│   (200MB)       │───▶│   Strategy      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   Big Files     │    │   Normal Files  │
                    │   (> 200MB)     │    │   (≤ 200MB)     │
                    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │  Sequential     │    │   Parallel      │
                    │  Processing     │    │   Processing    │
                    │  (1 worker)     │    │   (N workers)   │
                    └─────────────────┘    └─────────────────┘
                                │                       │
                                └───────┬───────────────┘
                                        ▼
                    ┌─────────────────────────────────────┐
                    │         Combined Results            │
                    │     (Sequential + Parallel)        │
                    └─────────────────────────────────────┘
```

## 📈 **Benefits**

### **✅ Optimal Resource Usage**
- **Big files** → Sequential (avoids memory issues)
- **Normal files** → Parallel (maximizes CPU usage)

### **✅ Memory Management**
- Large files processed one at a time
- Prevents memory overflow with big files
- Efficient for mixed file sizes

### **✅ Performance Optimization**
- Small files benefit from parallel processing
- Big files don't block the queue
- Better overall throughput

## 🎯 **Example Scenarios**

### **Scenario 1: Mixed File Sizes**
```
Files to process:
- song1.mp3 (5MB)     → Normal file → Parallel
- song2.mp3 (8MB)     → Normal file → Parallel  
- album.flac (250MB)  → Big file   → Sequential
- song3.mp3 (3MB)     → Normal file → Parallel

Processing order:
1. album.flac (Sequential, 1 worker)
2. song1.mp3, song2.mp3, song3.mp3 (Parallel, N workers)
```

### **Scenario 2: All Small Files**
```
Files to process:
- song1.mp3 (5MB)  → Normal file → Parallel
- song2.mp3 (8MB)  → Normal file → Parallel
- song3.mp3 (3MB)  → Normal file → Parallel

Processing: All files processed in parallel
```

### **Scenario 3: All Big Files**
```
Files to process:
- album1.flac (300MB) → Big file → Sequential
- album2.flac (250MB) → Big file → Sequential

Processing: All files processed sequentially (one at a time)
```

## 🔧 **Configuration**

### **Threshold Setting**
```python
BIG_FILE_SIZE_MB = 200  # 200MB threshold
```

### **Worker Configuration**
```python
# Sequential processing (big files)
workers = 1

# Parallel processing (normal files)  
workers = args.workers or max(1, mp.cpu_count())
```

## 🗄️ **Database Integration**

### **File Size Storage**
File sizes are stored in the `file_discovery_state` table during file discovery:

```sql
CREATE TABLE file_discovery_state (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT,
    file_size INTEGER,        -- File size in bytes
    last_modified REAL,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status TEXT DEFAULT 'active'
)
```

### **Size Retrieval**
```python
# Get file sizes from database instead of filesystem
file_sizes = audio_db.get_file_sizes_from_db(file_paths_only)

# Use stored sizes for classification
for file_path in file_paths_only:
    file_size_bytes = file_sizes.get(file_path, 0)
    file_size_mb = file_size_bytes / (1024 * 1024)
    # ... classification logic
```

### **Benefits**
- ✅ **Performance**: No filesystem I/O during analysis
- ✅ **Consistency**: Same size data used across runs
- ✅ **Efficiency**: Database queries are faster than stat() calls
- ✅ **Reliability**: No filesystem access errors during processing

## 📊 **Logging Output**

```
File distribution: 15 normal files, 2 big files (>200MB)
Processing 2 big files sequentially...
Processing 15 normal files in parallel...
```

## ✅ **Verification**

The new logic ensures:
- ✅ Files > 200MB are processed sequentially
- ✅ Files ≤ 200MB are processed in parallel
- ✅ Failed files are excluded from processing
- ✅ Optimal resource usage for different file sizes
- ✅ Clear logging of processing strategy

This approach provides the best balance of performance and resource management! 🚀 