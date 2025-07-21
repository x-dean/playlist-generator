import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

class AudioDatabase:
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.conn = None  # Initialize as None
        self.connect()  # Connect on initialization

    def connect(self):
        if os.path.exists(self.cache_file):
            self.conn = sqlite3.connect(self.cache_file, timeout=600)
            return True
        return False

    def get_all_features(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
            SELECT file_path, duration, bpm, beat_confidence, centroid,
                   loudness, danceability, key, scale, onset_rate, zcr
            FROM audio_features
            """)
            return [{
                'filepath': row[0],
                'duration': row[1],
                'bpm': row[2],
                'beat_confidence': row[3],
                'centroid': row[4],
                'loudness': row[5],
                'danceability': row[6],
                'key': row[7],
                'scale': row[8],
                'onset_rate': row[9],
                'zcr': row[10]
            } for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching features: {str(e)}")
            return []

    def cleanup_database(self):
        """Remove entries for files that no longer exist"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT file_path FROM audio_features")
            db_files = [row[0] for row in cursor.fetchall()]

            missing_files = [f for f in db_files if not os.path.exists(f)]

            if missing_files:
                logger.info(f"Cleaning up {len(missing_files)} missing files from database")
                placeholders = ','.join(['?'] * len(missing_files))
                cursor.execute(
                    f"DELETE FROM audio_features WHERE file_path IN ({placeholders})",
                    missing_files
                )
                self.conn.commit()
            return missing_files
        except Exception as e:
            logger.error(f"Database cleanup failed: {str(e)}")
            return []