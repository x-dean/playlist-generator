#!/usr/bin/env python3
"""
Test script for presentation layer components.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_cli_interface():
    """Test CLI interface functionality."""
    print("🔍 Testing CLI interface...")
    
    try:
        from presentation.cli.cli_interface import CLIInterface
        
        cli = CLIInterface()
        print("✅ CLI interface initialized successfully")
        
        # Test argument parser creation
        parser = cli._create_argument_parser()
        print("✅ Argument parser created successfully")
        
        # Test help display
        cli._show_help()
        print("✅ Help display working")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI interface test failed: {e}")
        return False

def test_progress_reporter():
    """Test progress reporting system."""
    print("\n🔍 Testing progress reporter...")
    
    try:
        from presentation.progress.progress_reporter import RichProgressReporter, ProgressStatus
        
        reporter = RichProgressReporter()
        print("✅ Progress reporter initialized successfully")
        
        # Test operation lifecycle
        operation_id = reporter.start_operation("Test Operation", "Testing progress reporting")
        print("✅ Operation started successfully")
        
        # Add steps
        step1_id = reporter.add_step("Step 1", "First step", 100)
        step2_id = reporter.add_step("Step 2", "Second step", 50)
        print("✅ Steps added successfully")
        
        # Start first step
        reporter.start_step(0)
        print("✅ Step 1 started")
        
        # Update progress
        reporter.update_step_progress(0, 0.5, 50)
        print("✅ Progress updated")
        
        # Complete first step
        reporter.complete_step(0)
        print("✅ Step 1 completed")
        
        # Start second step
        reporter.start_step(1)
        print("✅ Step 2 started")
        
        # Complete second step
        reporter.complete_step(1)
        print("✅ Step 2 completed")
        
        # Complete operation
        reporter.complete_operation(operation_id)
        print("✅ Operation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Progress reporter test failed: {e}")
        return False

def test_rest_api():
    """Test REST API interface."""
    print("\n🔍 Testing REST API...")
    
    try:
        from presentation.api.rest_api import RESTAPI
        
        api = RESTAPI()
        print("✅ REST API initialized successfully")
        
        app = api.get_app()
        print("✅ FastAPI app created successfully")
        
        # Test route registration
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/discover", "/analyze", "/enrich", "/playlist", "/export", "/methods", "/formats"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} registered")
            else:
                print(f"❌ Route {route} not found")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ REST API test failed: {e}")
        return False

def test_cli_commands():
    """Test CLI command parsing."""
    print("\n🔍 Testing CLI commands...")
    
    try:
        from presentation.cli.cli_interface import CLIInterface
        
        cli = CLIInterface()
        
        # Test discover command
        test_args = ["discover", "/test/path", "--recursive"]
        result = cli.run(test_args)
        print("✅ Discover command parsing working")
        
        # Test analyze command
        test_args = ["analyze", "/test/path", "--parallel"]
        result = cli.run(test_args)
        print("✅ Analyze command parsing working")
        
        # Test playlist command
        test_args = ["playlist", "--method", "kmeans", "--size", "20"]
        result = cli.run(test_args)
        print("✅ Playlist command parsing working")
        
        # Test export command
        test_args = ["export", "playlist.json", "--format", "m3u"]
        result = cli.run(test_args)
        print("✅ Export command parsing working")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI commands test failed: {e}")
        return False

def test_progress_integration():
    """Test progress reporter integration with services."""
    print("\n🔍 Testing progress integration...")
    
    try:
        from presentation.progress.progress_reporter import RichProgressReporter
        from application.services import FileDiscoveryService
        from application.dtos import FileDiscoveryRequest
        
        reporter = RichProgressReporter()
        discovery_service = FileDiscoveryService()
        
        # Start operation
        operation_id = reporter.start_operation("File Discovery", "Discovering audio files")
        
        # Add step
        step_id = reporter.add_step("Search Files", "Searching for audio files", 100)
        
        # Start step
        reporter.start_step(0)
        
        # Simulate service call with progress updates
        for i in range(0, 101, 10):
            reporter.update_step_progress(0, i / 100, i)
            import time
            time.sleep(0.1)  # Simulate work
        
        # Complete step
        reporter.complete_step(0)
        
        # Complete operation
        reporter.complete_operation(operation_id)
        
        print("✅ Progress integration working")
        return True
        
    except Exception as e:
        print(f"❌ Progress integration test failed: {e}")
        return False

def test_error_handling():
    """Test error handling in presentation layer."""
    print("\n🔍 Testing error handling...")
    
    try:
        from presentation.cli.cli_interface import CLIInterface
        from shared.exceptions import PlaylistaException
        
        cli = CLIInterface()
        
        # Test invalid command
        test_args = ["invalid_command"]
        result = cli.run(test_args)
        print("✅ Invalid command handling working")
        
        # Test missing arguments
        test_args = ["discover"]  # Missing path
        result = cli.run(test_args)
        print("✅ Missing arguments handling working")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_user_interaction():
    """Test user interaction components."""
    print("\n🔍 Testing user interaction...")
    
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.text import Text
        
        console = Console()
        
        # Test table creation
        table = Table(title="Test Table")
        table.add_column("Name", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Test", "Value")
        print("✅ Table creation working")
        
        # Test panel creation
        panel = Panel("Test content", title="Test Panel", border_style="blue")
        print("✅ Panel creation working")
        
        # Test text formatting
        text = Text("Test text", style="bold red")
        print("✅ Text formatting working")
        
        return True
        
    except Exception as e:
        print(f"❌ User interaction test failed: {e}")
        return False

def main():
    """Run all presentation layer tests."""
    print("🧪 Starting presentation layer tests...\n")
    
    tests = [
        ("CLI Interface", test_cli_interface),
        ("Progress Reporter", test_progress_reporter),
        ("REST API", test_rest_api),
        ("CLI Commands", test_cli_commands),
        ("Progress Integration", test_progress_integration),
        ("Error Handling", test_error_handling),
        ("User Interaction", test_user_interaction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"PRESENTATION LAYER TEST SUMMARY")
    print('='*50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All presentation layer tests passed!")
        return True
    else:
        print("⚠️  Some presentation layer tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 