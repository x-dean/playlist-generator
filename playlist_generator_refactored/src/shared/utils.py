"""
Shared utilities for the Playlista application.

This module provides common utilities used across the application,
including file handling, validation, and performance monitoring.
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import psutil

from shared.config import get_config


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


class UnifiedValidator:
    """Unified validation system for files and data."""
    
    def __init__(self):
        """Initialize the validator."""
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
    
    def validate_audio_file(self, file_path: Path) -> ValidationResult:
        """Validate an audio file comprehensively."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Basic file existence check
            if not file_path.exists():
                result.is_valid = False
                result.errors.append(f"File does not exist: {file_path}")
                return result
            
            # File type check
            if not self._is_valid_audio_extension(file_path):
                result.is_valid = False
                result.errors.append(f"Invalid audio file extension: {file_path.suffix}")
                return result
            
            # File size check
            file_size = file_path.stat().st_size
            if file_size < self.config.unified_processing.min_file_size_bytes:
                result.is_valid = False
                result.errors.append(f"File too small: {file_size} bytes")
                return result
            
            # File readability check
            if not self._is_file_readable(file_path):
                result.is_valid = False
                result.errors.append(f"File not readable: {file_path}")
                return result
            
            # Audio format validation (basic)
            if not self._is_valid_audio_format(file_path):
                result.warnings.append(f"Potentially invalid audio format: {file_path}")
            
            # Metadata validation
            metadata_result = self._validate_audio_metadata(file_path)
            if not metadata_result.is_valid:
                result.warnings.extend(metadata_result.warnings)
            
            # Store metadata
            result.metadata = {
                'file_size': file_size,
                'file_size_mb': file_size / (1024 * 1024),
                'extension': file_path.suffix.lower(),
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                'metadata_valid': metadata_result.is_valid
            }
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation error: {e}")
        
        return result
    
    def validate_file_paths(self, file_paths: List[str]) -> Dict[str, ValidationResult]:
        """Validate multiple file paths."""
        results = {}
        
        for file_path_str in file_paths:
            try:
                file_path = Path(file_path_str)
                results[file_path_str] = self.validate_audio_file(file_path)
            except Exception as e:
                results[file_path_str] = ValidationResult(
                    is_valid=False,
                    errors=[f"Path validation error: {e}"]
                )
        
        return results
    
    def _is_valid_audio_extension(self, file_path: Path) -> bool:
        """Check if file has a valid audio extension."""
        valid_extensions = self.config.unified_processing.get_audio_extensions_set()
        return file_path.suffix.lower() in valid_extensions
    
    def _is_file_readable(self, file_path: Path) -> bool:
        """Check if file is readable."""
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Read first 1KB to test readability
            return True
        except Exception:
            return False
    
    def _is_valid_audio_format(self, file_path: Path) -> bool:
        """Basic audio format validation."""
        try:
            # Try to read file header
            with open(file_path, 'rb') as f:
                header = f.read(12)
            
            # Check for common audio file signatures
            if file_path.suffix.lower() == '.mp3':
                return header.startswith(b'ID3') or header.startswith(b'\xff\xfb')
            elif file_path.suffix.lower() == '.flac':
                return header.startswith(b'fLaC')
            elif file_path.suffix.lower() == '.wav':
                return header.startswith(b'RIFF')
            elif file_path.suffix.lower() == '.m4a':
                return b'ftyp' in header
            else:
                # For other formats, assume valid if readable
                return True
                
        except Exception:
            return False
    
    def _validate_audio_metadata(self, file_path: Path) -> ValidationResult:
        """Validate audio file metadata."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Try to extract basic metadata
            import mutagen
            
            audio = mutagen.File(str(file_path))
            if audio is None:
                result.warnings.append("Could not read audio metadata")
                return result
            
            # Check for basic metadata fields
            if hasattr(audio, 'info'):
                info = audio.info
                if hasattr(info, 'length') and info.length == 0:
                    result.warnings.append("Audio file has zero duration")
                
                if hasattr(info, 'bitrate') and info.bitrate == 0:
                    result.warnings.append("Audio file has zero bitrate")
            
        except ImportError:
            result.warnings.append("mutagen not available for metadata validation")
        except Exception as e:
            result.warnings.append(f"Metadata validation error: {e}")
        
        return result


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except Exception:
        return 0.0


def sort_files_by_size(file_paths: List[Path], reverse: bool = False) -> List[Path]:
    """Sort files by size."""
    return sorted(file_paths, key=lambda p: p.stat().st_size, reverse=reverse)


def split_files_by_size(file_paths: List[Path], threshold_mb: float) -> Tuple[List[Path], List[Path]]:
    """Split files into small and large based on size threshold."""
    small_files = []
    large_files = []
    
    for file_path in file_paths:
        try:
            size_mb = get_file_size_mb(file_path)
            if size_mb < threshold_mb:
                small_files.append(file_path)
            else:
                large_files.append(file_path)
        except Exception:
            # If we can't get size, assume it's large
            large_files.append(file_path)
    
    return small_files, large_files


def log_largest_files(file_paths: List[Path], count: int = 5):
    """Log the largest files in the list."""
    if not file_paths:
        return
    
    sorted_files = sort_files_by_size(file_paths, reverse=True)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Largest {min(count, len(sorted_files))} files:")
    for i, file_path in enumerate(sorted_files[:count]):
        size_mb = get_file_size_mb(file_path)
        logger.info(f"  {i+1}. {file_path.name} ({size_mb:.1f}MB)")


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    try:
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    except ImportError:
        return {
            'total_gb': 0.0,
            'available_gb': 0.0,
            'used_gb': 0.0,
            'percent': 0.0
        }


def check_memory_pressure(threshold_percent: float = 80.0) -> bool:
    """Check if system is under memory pressure."""
    try:
        memory = psutil.virtual_memory()
        return memory.percent > threshold_percent
    except ImportError:
        return False


def calculate_file_hash(file_path: Path, algorithm: str = 'md5') -> Optional[str]:
    """Calculate file hash."""
    try:
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception:
        return None


def get_file_metadata(file_path: Path) -> Dict[str, Any]:
    """Get basic file metadata."""
    try:
        stat = file_path.stat()
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime),
            'modified': datetime.fromtimestamp(stat.st_mtime),
            'extension': file_path.suffix.lower(),
            'name': file_path.name,
            'path': str(file_path)
        }
    except Exception:
        return {}


# Global validator instance
_validator: Optional[UnifiedValidator] = None


def get_validator() -> UnifiedValidator:
    """Get the global validator instance."""
    global _validator
    if _validator is None:
        _validator = UnifiedValidator()
    return _validator


def validate_audio_files(file_paths: List[str]) -> Dict[str, ValidationResult]:
    """Validate multiple audio files."""
    validator = get_validator()
    return validator.validate_file_paths(file_paths) 