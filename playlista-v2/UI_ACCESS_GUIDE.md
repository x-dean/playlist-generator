# 🌐 Playlista v2 - UI Access Guide

## 📡 **Access Points**

### **🔗 Web Application URLs**
```
🌐 Web UI:           http://localhost:3000
📊 API Backend:      http://localhost:8000  
📚 API Docs:         http://localhost:8000/docs
❤️ Health Check:     http://localhost:8000/api/health
🗄️ Database:         localhost:5432 (playlista_v2)
🔴 Redis Cache:      localhost:6379
```

### **🎯 Current Status**
✅ **Frontend**: Running on port 3000 (Vite dev server)  
✅ **Backend**: Running on port 8000 (FastAPI)  
✅ **Database**: PostgreSQL with tables created  
✅ **Redis**: Cache server running  
✅ **Health Check**: API responding correctly  

## 🌐 **How to Use the Web UI**

### **1. Open Web Interface**
```bash
# Open in your browser:
http://localhost:3000
```

### **2. Web UI Features Available**
The React frontend provides:

🏠 **Dashboard** - Overview and stats  
📚 **Library** - Browse your music collection  
🔬 **Analysis** - Audio analysis tools  
🎵 **Playlists** - Generate and manage playlists  
📊 **Track Details** - Detailed track information  

### **3. Navigation**
- **Header Navigation**: Click sections to navigate
- **Modern UI**: Mantine component library
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: WebSocket integration ready

## 🗄️ **Database Content**

### **📊 Current Database Schema**
```sql
Tables Created:
✅ tracks      - Music track information
✅ playlists   - Generated playlists
```

### **🔍 Check Database Content**
```bash
# Connect to database
docker-compose exec postgres psql -U playlista -d playlista_v2

# List tables
\dt

# Check tracks
SELECT COUNT(*) FROM tracks;

# Check playlists  
SELECT COUNT(*) FROM playlists;

# Exit
\q
```

### **📋 Sample Queries**
```sql
-- View all tracks
SELECT id, title, artist, duration FROM tracks LIMIT 5;

-- View track features
SELECT title, audio_features FROM tracks WHERE audio_features IS NOT NULL;

-- View playlists
SELECT name, algorithm, track_count FROM playlists;
```

## 📊 **API Documentation**

### **🔗 Interactive API Docs**
```
Open: http://localhost:8000/docs
```

This provides:
- 📖 Complete API documentation
- 🧪 Interactive testing interface  
- 📝 Request/response examples
- 🔧 Parameter descriptions

### **🎯 Key API Endpoints**

#### **Library Management**
```http
GET  /api/library/tracks          # List all tracks
GET  /api/library/tracks/count    # Get track count
POST /api/analyze                 # Analyze audio file
```

#### **Playlist Generation**
```http  
POST /api/playlists/generate      # Generate new playlist
GET  /api/playlists               # List playlists
```

#### **System Health**
```http
GET  /api/health                  # System status
```

## 🧪 **Testing the Interface**

### **1. Test API Endpoints**
```powershell
# Health check
Invoke-WebRequest -Uri "http://localhost:8000/api/health"

# Get tracks (may be empty initially)
Invoke-WebRequest -Uri "http://localhost:8000/api/library/tracks"
```

### **2. Test Web UI**
1. Open `http://localhost:3000`
2. Navigate through sections
3. Try uploading/analyzing music
4. Generate test playlists

### **3. Populate with Real Data**
Run analysis on your music:
```bash
# Run comprehensive analysis
docker-compose exec backend python /app/test_full.py

# This will populate the database with real track data
```

## 📱 **Web UI Sections Explained**

### **🏠 Dashboard Page**
- **Purpose**: Overview of your music library
- **Features**: Stats, recent activity, quick actions
- **URL**: `http://localhost:3000/`

### **📚 Library Page**  
- **Purpose**: Browse and search your music collection
- **Features**: Track listing, filtering, metadata display
- **URL**: `http://localhost:3000/library`

### **🔬 Analysis Page**
- **Purpose**: Audio analysis tools and results
- **Features**: Feature visualization, analysis jobs
- **URL**: `http://localhost:3000/analysis`

### **🎵 Playlists Page**
- **Purpose**: Playlist generation and management
- **Features**: Algorithm selection, playlist creation
- **URL**: `http://localhost:3000/playlists`

## 🎯 **Next Steps**

### **To Populate Database:**
1. Run full music analysis (will take ~86 minutes)
2. Database will be populated with 86 tracks
3. Web UI will show real data

### **To Customize:**
1. Modify React components in `frontend/src/`
2. Add new API endpoints in `backend/app/api/`
3. Customize database schema in `backend/app/database/models.py`

## 🚀 **Performance Notes**

- **Frontend**: Hot reload enabled for development
- **Backend**: FastAPI with async processing
- **Database**: Optimized PostgreSQL with indexing
- **Real-time**: WebSocket support for live updates

## 🔧 **Troubleshooting**

### **Web UI Not Loading**
```bash
# Check frontend status
docker-compose ps frontend

# Restart frontend
docker-compose restart frontend
```

### **API Not Responding**
```bash
# Check backend logs
docker-compose logs backend --tail=20

# Restart backend
docker-compose restart backend
```

### **Database Issues**
```bash  
# Check database connection
docker-compose exec postgres pg_isready -U playlista

# View database logs
docker-compose logs postgres --tail=10
```

---

🎵 **Your Playlista v2 system is now ready to use!**  
Open http://localhost:3000 to start exploring your music collection.
