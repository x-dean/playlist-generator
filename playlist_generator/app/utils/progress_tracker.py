"""
Improved progress tracking for parallel audio analysis.
"""
import time
import threading
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProgressItem:
    """Represents a single progress item."""
    file_path: str
    status: str  # 'pending', 'processing', 'completed', 'failed', 'skipped'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


class ProgressTracker:
    """Thread-safe progress tracker for parallel processing."""
    
    def __init__(self, total_files: int, description: str = "Processing"):
        self.total_files = total_files
        self.description = description
        self.items: Dict[str, ProgressItem] = {}
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        self.last_update_time = time.time()
        self.update_callbacks: List[Callable] = []
        
        # Statistics
        self.completed_count = 0
        self.failed_count = 0
        self.skipped_count = 0
        self.processing_count = 0
    
    def add_file(self, file_path: str):
        """Add a file to the progress tracker."""
        with self.lock:
            if file_path not in self.items:
                self.items[file_path] = ProgressItem(
                    file_path=file_path,
                    status='pending'
                )
    
    def start_processing(self, file_path: str):
        """Mark a file as processing."""
        with self.lock:
            if file_path in self.items:
                self.items[file_path].status = 'processing'
                self.items[file_path].start_time = datetime.now()
                self.processing_count += 1
                self._trigger_update()
    
    def complete_file(self, file_path: str, success: bool = True, error_message: str = None):
        """Mark a file as completed."""
        with self.lock:
            if file_path in self.items:
                item = self.items[file_path]
                item.end_time = datetime.now()
                item.processing_time = (item.end_time - item.start_time).total_seconds() if item.start_time else None
                
                if success:
                    item.status = 'completed'
                    self.completed_count += 1
                else:
                    item.status = 'failed'
                    item.error_message = error_message
                    self.failed_count += 1
                
                self.processing_count -= 1
                self._trigger_update()
    
    def skip_file(self, file_path: str, reason: str = None):
        """Mark a file as skipped."""
        with self.lock:
            if file_path in self.items:
                self.items[file_path].status = 'skipped'
                self.items[file_path].error_message = reason
                self.skipped_count += 1
                self._trigger_update()
    
    def get_progress(self) -> Dict:
        """Get current progress statistics."""
        with self.lock:
            total_processed = self.completed_count + self.failed_count + self.skipped_count
            progress_percent = (total_processed / self.total_files * 100) if self.total_files > 0 else 0
            
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            
            # Calculate estimated time remaining
            if total_processed > 0 and elapsed_time > 0:
                rate = total_processed / elapsed_time
                remaining_files = self.total_files - total_processed
                eta_seconds = remaining_files / rate if rate > 0 else 0
            else:
                eta_seconds = 0
            
            return {
                'total_files': self.total_files,
                'completed': self.completed_count,
                'failed': self.failed_count,
                'skipped': self.skipped_count,
                'processing': self.processing_count,
                'pending': self.total_files - total_processed - self.processing_count,
                'progress_percent': progress_percent,
                'elapsed_time': elapsed_time,
                'eta_seconds': eta_seconds,
                'rate': total_processed / elapsed_time if elapsed_time > 0 else 0
            }
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """Get recent activity for display."""
        with self.lock:
            # Sort items by end_time (most recent first)
            sorted_items = sorted(
                [item for item in self.items.values() if item.end_time],
                key=lambda x: x.end_time,
                reverse=True
            )
            
            recent_activity = []
            for item in sorted_items[:limit]:
                recent_activity.append({
                    'file_path': item.file_path,
                    'status': item.status,
                    'processing_time': item.processing_time,
                    'error_message': item.error_message
                })
            
            return recent_activity
    
    def get_failed_files(self) -> List[str]:
        """Get list of failed files."""
        with self.lock:
            return [file_path for file_path, item in self.items.items() 
                   if item.status == 'failed']
    
    def add_update_callback(self, callback: Callable):
        """Add a callback to be called when progress updates."""
        with self.lock:
            self.update_callbacks.append(callback)
    
    def _trigger_update(self):
        """Trigger update callbacks."""
        current_time = time.time()
        # Throttle updates to avoid too frequent callbacks
        if current_time - self.last_update_time > 0.5:  # Update every 0.5 seconds
            self.last_update_time = current_time
            progress = self.get_progress()
            
            for callback in self.update_callbacks:
                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Error in progress callback: {e}")


class ParallelProgressTracker:
    """Specialized progress tracker for parallel processing."""
    
    def __init__(self, total_files: int, num_workers: int):
        self.tracker = ProgressTracker(total_files, "Parallel Processing")
        self.num_workers = num_workers
        self.worker_status = {}  # Track which worker is processing which file
        self.worker_lock = threading.Lock()
    
    def assign_worker(self, worker_id: int, file_path: str):
        """Assign a file to a worker."""
        with self.worker_lock:
            self.worker_status[worker_id] = file_path
            self.tracker.start_processing(file_path)
    
    def release_worker(self, worker_id: int, success: bool = True, error_message: str = None):
        """Release a worker and mark the file as completed."""
        with self.worker_lock:
            if worker_id in self.worker_status:
                file_path = self.worker_status[worker_id]
                self.tracker.complete_file(file_path, success, error_message)
                del self.worker_status[worker_id]
    
    def get_worker_status(self) -> Dict[int, str]:
        """Get current worker status."""
        with self.worker_lock:
            return dict(self.worker_status)
    
    def get_progress(self) -> Dict:
        """Get progress with worker information."""
        progress = self.tracker.get_progress()
        progress['active_workers'] = len(self.worker_status)
        progress['worker_status'] = self.get_worker_status()
        return progress


class ProgressDisplay:
    """Rich progress display for the CLI."""
    
    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker
        self.last_display_time = 0
        self.display_interval = 1.0  # Update display every second
    
    def should_update_display(self) -> bool:
        """Check if we should update the display."""
        current_time = time.time()
        if current_time - self.last_display_time >= self.display_interval:
            self.last_display_time = current_time
            return True
        return False
    
    def format_progress_bar(self, progress: Dict) -> str:
        """Format progress as a text bar."""
        completed = progress['completed']
        failed = progress['failed']
        skipped = progress['skipped']
        total = progress['total_files']
        
        # Calculate bar width
        bar_width = 40
        completed_width = int((completed / total) * bar_width) if total > 0 else 0
        failed_width = int((failed / total) * bar_width) if total > 0 else 0
        skipped_width = int((skipped / total) * bar_width) if total > 0 else 0
        
        # Build the bar
        bar = "█" * completed_width
        bar += "░" * failed_width
        bar += "▒" * skipped_width
        bar += " " * (bar_width - len(bar))
        
        return f"[{bar}] {progress['progress_percent']:.1f}%"
    
    def format_eta(self, eta_seconds: float) -> str:
        """Format estimated time remaining."""
        if eta_seconds <= 0:
            return "Unknown"
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def display_progress(self, progress: Dict):
        """Display progress information."""
        if not self.should_update_display():
            return
        
        bar = self.format_progress_bar(progress)
        eta = self.format_eta(progress['eta_seconds'])
        
        # Format processing rate
        rate = progress['rate']
        if rate > 0:
            rate_str = f"{rate:.1f} files/sec"
        else:
            rate_str = "N/A"
        
        # Display status
        status_line = (
            f"{bar} | "
            f"Completed: {progress['completed']} | "
            f"Failed: {progress['failed']} | "
            f"Skipped: {progress['skipped']} | "
            f"Processing: {progress['processing']} | "
            f"ETA: {eta} | "
            f"Rate: {rate_str}"
        )
        
        print(f"\r{status_line}", end="", flush=True)
    
    def finalize_display(self):
        """Finalize the display with a newline."""
        print()  # Add newline after progress bar


# Global progress tracker instance
progress_tracker = None 