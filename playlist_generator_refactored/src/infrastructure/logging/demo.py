#!/usr/bin/env python3
"""
Demo script showing the clean structured logging approach.
"""

import time
from pathlib import Path
from infrastructure.logging.helpers import (
    get_database_logger,
    get_analysis_logger,
    get_performance_logger
)
from infrastructure.logging.logger import setup_logging


def demo_clean_logging():
    """Demonstrate the clean structured logging approach."""
    
    # Setup logging
    setup_logging()
    
    # Get specialized loggers
    db_logger = get_database_logger()
    analysis_logger = get_analysis_logger()
    perf_logger = get_performance_logger()
    
    print("🎯 Clean Structured Logging Demo")
    print("=" * 50)
    
    # Demo database operations
    print("\n📊 Database Operations:")
    print("-" * 30)
    
    # Save operation - clean and simple
    db_logger.save_operation(
        entity_type='audio_file',
        entity_id='123e4567-e89b-12d3-a456-426614174000',
        file_path='/music/song.mp3',
        file_size_mb=5.2,
        file_hash='abc123...'
    )
    
    # Find operation
    db_logger.find_operation(
        entity_type='audio_file',
        entity_id='123e4567-e89b-12d3-a456-426614174000',
        found=True
    )
    
    # Success/failure logging
    db_logger.operation_success(
        duration_ms=45,
        rows_affected=1
    )
    
    db_logger.operation_failed(
        error_type='DatabaseError',
        error_message='Connection timeout',
        duration_ms=5000
    )
    
    # Demo analysis operations
    print("\n🔍 Analysis Operations:")
    print("-" * 30)
    
    file_path = Path('/music/song.mp3')
    
    # Start analysis
    analysis_logger.start_analysis(
        file_path=file_path,
        file_size_mb=5.2,
        processing_mode='parallel'
    )
    
    # Analysis phases
    analysis_logger.analysis_phase(
        phase='metadata_extraction',
        file_path=file_path,
        duration_ms=150
    )
    
    analysis_logger.analysis_phase(
        phase='feature_extraction',
        file_path=file_path,
        duration_ms=2500
    )
    
    # Complete analysis
    analysis_logger.analysis_complete(
        file_path=file_path,
        duration_ms=3000,
        features_extracted=15,
        quality_score=0.85
    )
    
    # Demo performance monitoring
    print("\n⏱️ Performance Monitoring:")
    print("-" * 30)
    
    # Start timing an operation
    operation_id = 'file_processing_001'
    perf_logger.start_timer(
        operation_id=operation_id,
        operation_name='Process audio file',
        file_path='/music/song.mp3',
        file_size_mb=5.2
    )
    
    # Simulate work
    time.sleep(0.5)
    
    # End timing
    duration_ms = perf_logger.end_timer(
        operation_id=operation_id,
        success=True,
        features_extracted=15
    )
    
    print(f"✅ Operation took {duration_ms}ms")
    
    print("\n🎉 Demo completed! Check the log file for structured JSON output.")


if __name__ == '__main__':
    demo_clean_logging() 