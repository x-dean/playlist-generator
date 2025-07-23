import json
import os
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CheckpointManager:
    def __init__(self, checkpoint_dir: str = None):
        # Use cache directory from environment or default to 'checkpoints' in cache dir
        cache_dir = os.getenv('CACHE_DIR', '/app/cache')
        self.checkpoint_dir = checkpoint_dir or os.path.join(cache_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.current_checkpoint = None
        self.checkpoint_file = None
        logger.debug(f"Using checkpoint directory: {self.checkpoint_dir}")

    def save_checkpoint(self, state: Dict[str, Any], name: str = None) -> str:
        """Save processing state for recovery"""
        try:
            if name is None:
                name = f"checkpoint_{int(time.time())}"
            
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'name': name,
                'state': state
            }
            
            filepath = os.path.join(self.checkpoint_dir, f"{name}.json")
            with open(filepath, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.current_checkpoint = name
            self.checkpoint_file = filepath
            logger.debug(f"Saved checkpoint: {filepath}")
            return filepath
        
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            return None

    def load_checkpoint(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load last known good state or specific checkpoint"""
        try:
            if name is None:
                # Find most recent checkpoint
                checkpoints = self._get_checkpoints()
                if not checkpoints:
                    return None
                name = checkpoints[-1]['name']
            
            filepath = os.path.join(self.checkpoint_dir, f"{name}.json")
            if not os.path.exists(filepath):
                logger.warning(f"Checkpoint not found: {filepath}")
                return None
            
            with open(filepath, 'r') as f:
                checkpoint_data = json.load(f)
            
            self.current_checkpoint = name
            self.checkpoint_file = filepath
            logger.info(f"Loaded checkpoint: {filepath}")
            return checkpoint_data['state']
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None

    def _get_checkpoints(self) -> list:
        """Get list of available checkpoints, sorted by timestamp"""
        checkpoints = []
        try:
            for filename in os.listdir(self.checkpoint_dir):
                if not filename.endswith('.json'):
                    continue
                
                filepath = os.path.join(self.checkpoint_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        checkpoints.append({
                            'name': data['name'],
                            'timestamp': data['timestamp'],
                            'filepath': filepath
                        })
                except Exception as e:
                    logger.warning(f"Error reading checkpoint {filepath}: {str(e)}")
            
            return sorted(checkpoints, key=lambda x: x['timestamp'])
        
        except Exception as e:
            logger.error(f"Error listing checkpoints: {str(e)}")
            return []

    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """Remove checkpoints older than max_age_hours"""
        removed = 0
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            for checkpoint in self._get_checkpoints():
                try:
                    timestamp = datetime.fromisoformat(checkpoint['timestamp']).timestamp()
                    if current_time - timestamp > max_age_seconds:
                        os.remove(checkpoint['filepath'])
                        removed += 1
                except Exception as e:
                    logger.warning(f"Error removing checkpoint {checkpoint['filepath']}: {str(e)}")
            
            if removed > 0:
                logger.info(f"Removed {removed} old checkpoints")
            return removed
        
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {str(e)}")
            return 0

    def get_recovery_state(self) -> Optional[Dict[str, Any]]:
        """Get the most recent valid checkpoint state"""
        checkpoints = self._get_checkpoints()
        
        for checkpoint in reversed(checkpoints):
            try:
                with open(checkpoint['filepath'], 'r') as f:
                    data = json.load(f)
                    if self._validate_checkpoint(data):
                        logger.info(f"Found valid recovery state from {data['timestamp']}")
                        return data['state']
            except Exception as e:
                logger.warning(f"Error validating checkpoint {checkpoint['filepath']}: {str(e)}")
                continue
        
        return None

    def _validate_checkpoint(self, data: Dict[str, Any]) -> bool:
        """Validate checkpoint data structure"""
        required_fields = ['timestamp', 'name', 'state']
        return all(field in data for field in required_fields) 