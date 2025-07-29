"""
Examples of how to use the structured logging system.
"""

from pathlib import Path
from infrastructure.logging.helpers import (
    get_database_logger,
    get_analysis_logger,
    get_performance_logger
)


def example_database_logging():
    """Example of database logging."""
    db_logger = get_database_logger()
    
    # Simple save operation
    db_logger.save_operation(
        entity_type='audio_file',
        entity_id='123e4567-e89b-12d3-a456-426614174000',
        file_path='/music/song.mp3',
        file_size_mb=5.2
    )
    
    # Query operation
    db_logger.query_operation(
        query_type='SELECT',
        table='audio_files',
        limit=100,
        offset=0
    )
    
    # Operation with timing
    import time
    start_time = time.time()
    
    # ... do database work ...
    time.sleep(0.1)  # Simulate work
    
    duration_ms = int((time.time() - start_time) * 1000)
    db_logger.operation_success(
        duration_ms=duration_ms,
        rows_affected=1
    )


def example_analysis_logging():
    """Example of analysis logging."""
    analysis_logger = get_analysis_logger()
    
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


def example_performance_logging():
    """Example of performance logging."""
    perf_logger = get_performance_logger()
    
    # Start timing an operation
    operation_id = 'file_processing_001'
    perf_logger.start_timer(
        operation_id=operation_id,
        operation_name='Process audio file',
        file_path='/music/song.mp3',
        file_size_mb=5.2
    )
    
    # ... do work ...
    import time
    time.sleep(0.5)  # Simulate work
    
    # End timing
    duration_ms = perf_logger.end_timer(
        operation_id=operation_id,
        success=True,
        features_extracted=15
    )
    
    print(f"Operation took {duration_ms}ms")


def example_clean_database_operations():
    """Example of clean database operations with structured logging."""
    db_logger = get_database_logger()
    
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
    
    # Delete operation
    db_logger.delete_operation(
        entity_type='audio_file',
        entity_id='123e4567-e89b-12d3-a456-426614174000',
        cascading=True
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


if __name__ == '__main__':
    # Run examples
    example_database_logging()
    example_analysis_logging()
    example_performance_logging()
    example_clean_database_operations() 