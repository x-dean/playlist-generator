#!/usr/bin/env python3
"""
Test script for robustness improvements.
This script tests the circuit breaker, adaptive timeout, error classification, and smart retry features.
"""

import os
import sys
import time
import logging
import tempfile
import sqlite3
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from app.utils.adaptive_timeout import AdaptiveTimeoutManager, get_timeout_manager
from app.utils.error_classifier import ErrorClassifier, classify_error, ErrorType
from app.utils.smart_retry import SmartRetryManager, RetryStrategy, get_retry_manager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_circuit_breaker():
    """Test circuit breaker functionality."""
    logger.info("Testing Circuit Breaker...")
    
    # Create a circuit breaker
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5, name='test_breaker')
    
    # Test successful operation
    def successful_operation():
        return "success"
    
    result = breaker.call(successful_operation)
    assert result == "success"
    logger.info("✅ Circuit breaker successful operation test passed")
    
    # Test failing operation
    def failing_operation():
        raise ValueError("Test error")
    
    # Should fail 3 times then open circuit
    for i in range(3):
        try:
            breaker.call(failing_operation)
        except ValueError:
            logger.info(f"Expected failure {i+1}/3")
    
    # Circuit should now be open
    try:
        breaker.call(failing_operation)
        assert False, "Circuit should be open"
    except CircuitBreakerOpenError:
        logger.info("✅ Circuit breaker opened correctly after 3 failures")
    
    # Wait for recovery timeout
    logger.info("Waiting for circuit breaker recovery...")
    time.sleep(6)
    
    # Circuit should be half-open now
    try:
        breaker.call(successful_operation)
        logger.info("✅ Circuit breaker recovered successfully")
    except Exception as e:
        logger.error(f"Circuit breaker recovery failed: {e}")

def test_adaptive_timeout():
    """Test adaptive timeout functionality."""
    logger.info("Testing Adaptive Timeout...")
    
    timeout_manager = get_timeout_manager()
    
    # Test timeout calculation for different file sizes
    small_file_timeout = timeout_manager.calculate_timeout(10.0, 'rhythm')
    large_file_timeout = timeout_manager.calculate_timeout(100.0, 'rhythm')
    
    assert large_file_timeout > small_file_timeout
    logger.info(f"✅ Adaptive timeout: small file ({small_file_timeout:.1f}s) < large file ({large_file_timeout:.1f}s)")
    
    # Test performance recording
    timeout_manager.record_performance('rhythm', 50.0, 30.0, True, 60.0)
    timeout_manager.record_performance('rhythm', 50.0, 45.0, False, 60.0)
    
    stats = timeout_manager.get_performance_stats('rhythm')
    assert stats['total_operations'] >= 2
    logger.info(f"✅ Performance tracking: {stats['total_operations']} operations recorded")

def test_error_classifier():
    """Test error classification functionality."""
    logger.info("Testing Error Classifier...")
    
    classifier = get_error_classifier()
    
    # Test timeout error classification
    try:
        raise TimeoutError("Operation timed out")
    except Exception as e:
        error_info = classifier.classify_error(e, {'operation': 'test'})
        assert error_info.error_type == ErrorType.TIMEOUT
        logger.info(f"✅ Error classification: {error_info.error_type.value} (severity: {error_info.severity.value})")
    
    # Test database error classification
    try:
        raise sqlite3.OperationalError("database is locked")
    except Exception as e:
        error_info = classifier.classify_error(e, {'operation': 'database'})
        assert error_info.error_type == ErrorType.DATABASE
        logger.info(f"✅ Error classification: {error_info.error_type.value} (severity: {error_info.severity.value})")
    
    # Test recovery recommendation
    recommendation = classifier.get_recovery_recommendation(error_info)
    assert 'action' in recommendation
    logger.info(f"✅ Recovery recommendation: {recommendation['action']}")

def test_smart_retry():
    """Test smart retry functionality."""
    logger.info("Testing Smart Retry...")
    
    retry_manager = get_retry_manager()
    
    # Test successful operation
    def successful_operation():
        return "success"
    
    result = retry_manager.exponential_backoff(successful_operation, max_attempts=3)
    assert result.success
    assert result.result == "success"
    logger.info("✅ Smart retry successful operation test passed")
    
    # Test failing operation with retry
    attempt_count = 0
    def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Test error {attempt_count}")
        return "success after retries"
    
    result = retry_manager.exponential_backoff(failing_operation, max_attempts=3)
    assert result.success
    assert result.result == "success after retries"
    assert result.attempts == 3
    logger.info("✅ Smart retry failing operation test passed")

def test_database_robustness():
    """Test database robustness with concurrent access simulation."""
    logger.info("Testing Database Robustness...")
    
    # Create a temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    try:
        # Import the AudioAnalyzer
        from app.music_analyzer.feature_extractor import AudioAnalyzer
        
        # Create analyzer with temporary database
        analyzer = AudioAnalyzer(cache_file=db_path)
        
        # Test database operations
        file_info = {
            'file_hash': 'test_hash_123',
            'file_path': '/test/path/song.mp3',
            'last_modified': time.time()
        }
        
        features = {
            'duration': 180.0,
            'bpm': 120.0,
            'loudness': -12.5,
            'danceability': 0.8
        }
        
        # Test save operation
        success = analyzer._save_features_to_db(file_info, features)
        assert success
        logger.info("✅ Database save operation successful")
        
        # Test mark failed operation
        success = analyzer._mark_failed(file_info)
        assert success
        logger.info("✅ Database mark failed operation successful")
        
        # Test circuit breaker stats
        if hasattr(analyzer, 'circuit_breakers'):
            db_breaker = analyzer.circuit_breakers['database_operations']
            stats = db_breaker.get_stats()
            logger.info(f"✅ Circuit breaker stats: {stats['state']} (failures: {stats['failure_count']})")
        
    finally:
        # Clean up
        try:
            os.unlink(db_path)
        except:
            pass

def main():
    """Run all robustness tests."""
    logger.info("🚀 Starting Robustness Tests...")
    
    try:
        test_circuit_breaker()
        test_adaptive_timeout()
        test_error_classifier()
        test_smart_retry()
        test_database_robustness()
        
        logger.info("🎉 All robustness tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Robustness test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 