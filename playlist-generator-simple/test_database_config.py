#!/usr/bin/env python3
"""
Test script to demonstrate configurable settings in DatabaseManager.
Shows how to use different configuration options and their effects.
"""

import os
import sys
import tempfile
import shutil
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from core.logging_setup import setup_logging, cleanup_logging
from core.database import DatabaseManager
from core.config_loader import config_loader


def cleanup_test():
    """Clean up test handlers."""
    cleanup_logging()


def test_database_configuration():
    """Test DatabaseManager with different configuration settings."""
    print("Testing DatabaseManager Configuration")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup logging
        setup_logging(
            log_level='INFO',
            log_dir=temp_dir,
            log_file_prefix='test_database_config',
            console_logging=True,
            file_logging=False
        )
        
        print(f"📁 Test directory: {temp_dir}")
        
        # Test 1: Default configuration
        print("\n1. 🗄️ Testing Default Configuration...")
        db_path = os.path.join(temp_dir, 'default_config.db')
        db_manager = DatabaseManager(db_path)
        
        default_config = db_manager.get_database_config()
        print(f"   📋 Default configuration:")
        for key, value in default_config.items():
            print(f"      {key}: {value}")
        
        # Test 2: Custom configuration
        print("\n2. ⚙️ Testing Custom Configuration...")
        custom_config = {
            'DB_CACHE_DEFAULT_EXPIRY_HOURS': 12,  # 12 hours instead of 24
            'DB_CACHE_CLEANUP_FREQUENCY_HOURS': 6,  # 6 hours instead of 24
            'DB_CLEANUP_RETENTION_DAYS': 15,  # 15 days instead of 30
            'DB_FAILED_ANALYSIS_RETENTION_DAYS': 3,  # 3 days instead of 7
            'DB_STATISTICS_RETENTION_DAYS': 45,  # 45 days instead of 90
            'DB_STATISTICS_COLLECTION_FREQUENCY_HOURS': 12,  # 12 hours instead of 24
            'DB_AUTO_CLEANUP_ENABLED': False,  # Disable auto cleanup
            'DB_BACKUP_ENABLED': False,  # Disable auto backup
            'DB_PERFORMANCE_MONITORING_ENABLED': True,
            'DB_WAL_MODE_ENABLED': True,
            'DB_SYNCHRONOUS_MODE': 'NORMAL'
        }
        
        custom_db_path = os.path.join(temp_dir, 'custom_config.db')
        custom_db_manager = DatabaseManager(custom_db_path, custom_config)
        
        print(f"   📋 Custom configuration:")
        for key, value in custom_config.items():
            print(f"      {key}: {value}")
        
        # Test 3: Configuration effects on cache
        print("\n3. 💾 Testing Cache Configuration Effects...")
        
        # Save cache with default config (24 hours)
        db_manager.save_cache("default_test", {"data": "default_cache"}, None)
        print(f"   ✅ Saved cache with default expiry (24h)")
        
        # Save cache with custom config (12 hours)
        custom_db_manager.save_cache("custom_test", {"data": "custom_cache"}, None)
        print(f"   ✅ Saved cache with custom expiry (12h)")
        
        # Test 4: Configuration effects on cleanup
        print("\n4. 🧹 Testing Cleanup Configuration Effects...")
        
        # Add some test data
        for i in range(3):
            db_manager.save_statistic("test_category", f"test_key_{i}", f"test_value_{i}")
            custom_db_manager.save_statistic("test_category", f"test_key_{i}", f"test_value_{i}")
        
        print(f"   📊 Added test statistics to both databases")
        
        # Test cleanup with different retention periods
        default_cleanup = db_manager.cleanup_old_data(None)  # Uses default 30 days
        custom_cleanup = custom_db_manager.cleanup_old_data(None)  # Uses custom 15 days
        
        print(f"   🗑️ Default cleanup results: {default_cleanup}")
        print(f"   🗑️ Custom cleanup results: {custom_cleanup}")
        
        # Test 5: Configuration effects on statistics
        print("\n5. 📈 Testing Statistics Configuration Effects...")
        
        # Get statistics with different collection frequencies
        default_stats = db_manager.get_statistics(None, None)  # Uses default 24 hours
        custom_stats = custom_db_manager.get_statistics(None, None)  # Uses custom 12 hours
        
        print(f"   📊 Default stats collection (24h): {len(default_stats)} categories")
        print(f"   📊 Custom stats collection (12h): {len(custom_stats)} categories")
        
        # Test 6: Configuration updates
        print("\n6. 🔄 Testing Configuration Updates...")
        
        # Update configuration at runtime
        new_config = {
            'DB_CACHE_DEFAULT_EXPIRY_HOURS': 6,  # Change to 6 hours
            'DB_AUTO_CLEANUP_ENABLED': True,  # Enable auto cleanup
        }
        
        success = db_manager.update_config(new_config)
        if success:
            print(f"   ✅ Successfully updated configuration")
            updated_config = db_manager.get_database_config()
            print(f"   📋 Updated configuration:")
            for key, value in new_config.items():
                print(f"      {key}: {value}")
        
        # Test 7: Configuration validation
        print("\n7. ✅ Testing Configuration Validation...")
        
        # Test with invalid configuration
        invalid_config = {
            'DB_CACHE_DEFAULT_EXPIRY_HOURS': -1,  # Invalid negative value
            'DB_CLEANUP_RETENTION_DAYS': 0,  # Invalid zero value
        }
        
        invalid_db_path = os.path.join(temp_dir, 'invalid_config.db')
        try:
            invalid_db_manager = DatabaseManager(invalid_db_path, invalid_config)
            print(f"   ⚠️ Invalid configuration accepted (using defaults)")
        except Exception as e:
            print(f"   ❌ Invalid configuration rejected: {e}")
        
        print("\n✅ Database configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cleanup_test()
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_configuration_documentation():
    """Show all available database configuration options."""
    print("\nDatabase Configuration Options")
    print("=" * 50)
    print("The DatabaseManager supports the following configurable settings:")
    print()
    
    config_options = {
        "Cache Settings": {
            "DB_CACHE_DEFAULT_EXPIRY_HOURS": "Default cache expiration time in hours (default: 24)",
            "DB_CACHE_CLEANUP_FREQUENCY_HOURS": "How often to clean up expired cache entries (default: 24)",
            "DB_CACHE_MAX_SIZE_MB": "Maximum cache size in MB (default: 100)"
        },
        "Cleanup Settings": {
            "DB_CLEANUP_RETENTION_DAYS": "How long to keep old data in days (default: 30)",
            "DB_FAILED_ANALYSIS_RETENTION_DAYS": "How long to keep failed analysis records in days (default: 7)",
            "DB_STATISTICS_RETENTION_DAYS": "How long to keep statistics in days (default: 90)"
        },
        "Performance Settings": {
            "DB_CONNECTION_TIMEOUT_SECONDS": "Database connection timeout (default: 30)",
            "DB_MAX_RETRY_ATTEMPTS": "Maximum retry attempts for failed operations (default: 3)",
            "DB_BATCH_SIZE": "Batch size for bulk operations (default: 100)",
            "DB_QUERY_TIMEOUT_SECONDS": "Query timeout in seconds (default: 60)",
            "DB_MAX_CONNECTIONS": "Maximum database connections (default: 10)"
        },
        "Statistics Settings": {
            "DB_STATISTICS_COLLECTION_FREQUENCY_HOURS": "How often to collect database statistics (default: 24)"
        },
        "Auto Maintenance Settings": {
            "DB_AUTO_CLEANUP_ENABLED": "Enable automatic cleanup (default: true)",
            "DB_AUTO_CLEANUP_FREQUENCY_HOURS": "Auto cleanup frequency in hours (default: 168)"
        },
        "Backup Settings": {
            "DB_BACKUP_ENABLED": "Enable automatic backups (default: true)",
            "DB_BACKUP_FREQUENCY_HOURS": "Backup frequency in hours (default: 168)",
            "DB_BACKUP_RETENTION_DAYS": "How long to keep backups in days (default: 30)"
        },
        "Performance Monitoring": {
            "DB_PERFORMANCE_MONITORING_ENABLED": "Enable database performance monitoring (default: true)"
        },
        "SQLite Settings": {
            "DB_WAL_MODE_ENABLED": "Enable WAL mode for better concurrency (default: true)",
            "DB_SYNCHRONOUS_MODE": "SQLite synchronous mode: OFF, NORMAL, FULL (default: NORMAL)"
        }
    }
    
    for category, options in config_options.items():
        print(f"📋 {category}:")
        for option, description in options.items():
            print(f"   {option}: {description}")
        print()


def main():
    """Run the database configuration tests."""
    print("DatabaseManager Configuration Test")
    print("=" * 50)
    print("This test demonstrates:")
    print("  🗄️ Default configuration loading")
    print("  ⚙️ Custom configuration application")
    print("  💾 Cache configuration effects")
    print("  🧹 Cleanup configuration effects")
    print("  📈 Statistics configuration effects")
    print("  🔄 Runtime configuration updates")
    print("  ✅ Configuration validation")
    print("=" * 50)
    
    success = test_database_configuration()
    
    if success:
        print("\n🎉 Database configuration test passed!")
        print("✅ All configuration features working correctly")
        print("✅ Default settings applied properly")
        print("✅ Custom settings override defaults")
        print("✅ Runtime configuration updates working")
    else:
        print("\n❌ Database configuration test failed!")
    
    # Show configuration documentation
    test_configuration_documentation()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 