# File Discovery & Analysis Process Schematic

## 🔄 **Overall Process Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   File System   │    │  FileDiscovery  │    │   AudioAnalyzer │
│   (/music/*)    │───▶│   Module        │───▶│   (Database)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Audio Files    │    │  Change Tracking│    │  Analysis       │
│  Discovery      │    │  (DB State)     │    │  Results        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 **File Discovery Process**

```
┌─────────────────────────────────────────────────────────────────┐
│                    FileDiscovery.discover_files()              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Scan /music directory recursively                          │
│ 2. Skip /music/failed_files directory                        │
│ 3. Validate audio file extensions                             │
│ 4. Check file size (> 1KB)                                   │
│ 5. Return list of valid audio files                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Database Integration                        │
├─────────────────────────────────────────────────────────────────┤
│ • audio_features table: Analysis results                      │
│ • file_discovery_state table: File tracking                   │
│ • Status: 'active' or 'removed'                              │
│ • Tracks: file_path, hash, size, timestamps                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 **Analysis Mode Selection**

```
┌─────────────────────────────────────────────────────────────────┐
│                get_files_for_analysis()                       │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ Normal Mode │  │ Force Mode  │  │ Failed Mode │          │
│  │             │  │             │  │             │          │
│  │ • New files │  │ • All files │  │ • Failed    │          │
│  │ • Not in DB │  │ • Not failed│  │ • Not in    │          │
│  │ • Not failed│  │ • Re-analyze│  │   failed dir│          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 **Change Tracking Process**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Previous State  │    │ Current State   │    │ Change Detection│
│ (DB)            │    │ (Disk Scan)     │    │ (Comparison)    │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • file_path     │    │ • Discovered    │    │ • Added files   │
│ • status        │    │   files         │    │ • Removed files │
│ • last_seen_at  │    │ • File info     │    │ • Unchanged     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Database Update       │
                    │ • Mark old as 'removed'│
                    │ • Add new as 'active'  │
                    │ • Update timestamps    │
                    └─────────────────────────┘
```

## 🚀 **Processing Pipeline**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ File List   │───▶│ Processor   │───▶│ Results     │      │
│  │ (Validated) │    │ Selection   │    │ (Features)  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                   │                   │             │
│         ▼                   ▼                   ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Sequential  │    │ Parallel    │    │ Big Files   │      │
│  │ Processor   │    │ Processor   │    │ Processor   │      │
│  │ (< 10 files)│    │ (> 10 files)│    │ (> 100MB)  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🛡️ **Exclusion Logic**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Exclusion Checks                            │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  File Path ──┐                                                │
│              │                                                │
│              ▼                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │ In Failed Dir?  │  │ File Exists?    │  │ Valid Audio?    ││
│  │ /music/failed/  │  │ os.path.exists  │  │ Extension &     ││
│  │                 │  │                 │  │ Size Check      ││
│  └─────────┬───────┘  └─────────┬───────┘  └─────────┬───────┘│
│            │                     │                     │        │
│            ▼                     ▼                     ▼        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   EXCLUDE   │    │   EXCLUDE   │    │   EXCLUDE   │        │
│  │   (Skip)    │    │   (Skip)    │    │   (Skip)    │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 💾 **Database Schema**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Database Tables                              │
├─────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              audio_features                             │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ • file_hash (PRIMARY KEY)                             │   │
│  │ • file_path                                           │   │
│  │ • duration, bpm, centroid, loudness, etc.            │   │
│  │ • last_analyzed                                       │   │
│  │ • failed (0/1)                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              file_discovery_state                      │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ • file_path (PRIMARY KEY)                             │   │
│  │ • file_hash                                           │   │
│  │ • file_size                                           │   │
│  │ • last_modified                                       │   │
│  │ • discovered_at                                        │   │
│  │ • last_seen_at                                        │   │
│  │ • status ('active'/'removed')                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                               │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 **State Management Flow**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Start Run     │    │   File Scan     │    │   Update State  │
│                 │    │                 │    │                 │
│ • Load previous │    │ • Discover      │    │ • Mark old as   │
│   state from DB │    │   current files │    │   'removed'     │
│ • Get file list │    │ • Validate      │    │ • Mark new as   │
│ • Check changes │    │   files         │    │   'active'      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Analysis      │    │   Processing    │    │   Cleanup       │
│                 │    │                 │    │                 │
│ • Select files  │    │ • Extract       │    │ • Remove        │
│   for analysis  │    │   features      │    │   non-existent  │
│ • Choose mode   │    │ • Save to DB    │    │ • Update status │
│ • Feed to       │    │ • Update state  │    │                 │
│   processors    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎯 **Key Benefits**

✅ **Centralized State** - All file info in database  
✅ **Atomic Operations** - Database transactions ensure consistency  
✅ **Change Tracking** - Know what files were added/removed  
✅ **Exclusion Logic** - Consistent filtering across all components  
✅ **Performance** - No separate file I/O, efficient queries  
✅ **Cleanup** - Automatic removal of non-existent files  
✅ **Debugging** - Clear state visibility in database  

## 🔧 **Integration Points**

- **FileDiscovery** ↔ **AudioAnalyzer** - Database sharing
- **AnalysisManager** ↔ **FileDiscovery** - File list coordination  
- **Parallel/Sequential** ↔ **FileDiscovery** - Exclusion checks
- **FeatureExtractor** ↔ **FileDiscovery** - Validation logic

This schematic shows how the new database-integrated file discovery system provides a clean, centralized approach to managing audio file analysis with proper change tracking and exclusion logic. 