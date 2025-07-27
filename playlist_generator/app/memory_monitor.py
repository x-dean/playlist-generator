#!/usr/bin/env python3
"""
Memory monitoring script for playlist generator.
"""

import psutil
import time
import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor system memory usage and provide recommendations."""
    
    def __init__(self, critical_threshold=85, warning_threshold=75):
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold
    
    def get_memory_info(self):
        """Get detailed memory information."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_total_gb': memory.total / (1024**3),
                'swap_percent': swap.percent,
                'swap_used_gb': swap.used / (1024**3),
                'swap_total_gb': swap.total / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return None
    
    def get_cpu_info(self):
        """Get CPU information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            return cpu_percent, cpu_count
        except Exception as e:
            logger.error(f"Error getting CPU info: {e}")
            return 0, 0
    
    def get_process_info(self):
        """Get information about Python processes."""
        try:
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
                try:
                    if 'python' in proc.info['name'].lower():
                        cmdline = proc.info.get('cmdline', [])
                        if any('playlista' in arg for arg in cmdline):
                            memory_mb = proc.info['memory_info'].rss / (1024**2)
                            python_processes.append({
                                'pid': proc.info['pid'],
                                'memory_mb': memory_mb,
                                'cmdline': ' '.join(cmdline[:3]) + '...' if len(cmdline) > 3 else ' '.join(cmdline)
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return python_processes
        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return []
    
    def print_status(self):
        """Print current system status."""
        memory_info = self.get_memory_info()
        cpu_percent, cpu_count = self.get_cpu_info()
        processes = self.get_process_info()
        
        if not memory_info:
            print("❌ Could not get memory information")
            return
        
        print(f"\n📊 System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Memory status
        memory_status = "🟢 OK"
        if memory_info['memory_percent'] > self.critical_threshold:
            memory_status = "🔴 CRITICAL"
        elif memory_info['memory_percent'] > self.warning_threshold:
            memory_status = "🟡 WARNING"
        
        print(f"💾 Memory: {memory_status}")
        print(f"   Usage: {memory_info['memory_percent']:.1f}%")
        print(f"   Used: {memory_info['memory_used_gb']:.1f}GB")
        print(f"   Available: {memory_info['memory_available_gb']:.1f}GB")
        print(f"   Total: {memory_info['memory_total_gb']:.1f}GB")
        
        # Swap status
        if memory_info['swap_total_gb'] > 0:
            swap_status = "🟢 OK"
            if memory_info['swap_percent'] > 50:
                swap_status = "🟡 WARNING"
            if memory_info['swap_percent'] > 80:
                swap_status = "🔴 CRITICAL"
            
            print(f"💿 Swap: {swap_status}")
            print(f"   Usage: {memory_info['swap_percent']:.1f}%")
            print(f"   Used: {memory_info['swap_used_gb']:.1f}GB")
            print(f"   Total: {memory_info['swap_total_gb']:.1f}GB")
        
        # CPU status
        cpu_status = "🟢 OK"
        if cpu_percent > 80:
            cpu_status = "🟡 WARNING"
        if cpu_percent > 95:
            cpu_status = "🔴 CRITICAL"
        
        print(f"🖥️  CPU: {cpu_status}")
        print(f"   Usage: {cpu_percent:.1f}% of {cpu_count} cores")
        
        # Process information
        if processes:
            print(f"\n🐍 Python Processes ({len(processes)} found):")
            total_memory = 0
            for proc in processes:
                print(f"   PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['cmdline']}")
                total_memory += proc['memory_mb']
            print(f"   Total Python memory: {total_memory:.1f}MB")
        else:
            print("\n🐍 No Python processes found")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if memory_info['memory_percent'] > self.critical_threshold:
            print("   🔴 CRITICAL: Memory usage too high!")
            print("   - Stop other applications")
            print("   - Use --memory_aware flag")
            print("   - Use --workers=1 for sequential processing")
            print("   - Consider increasing swap space")
        elif memory_info['memory_percent'] > self.warning_threshold:
            print("   🟡 WARNING: Memory usage high")
            print("   - Consider using --memory_aware flag")
            print("   - Monitor for memory leaks")
        else:
            print("   🟢 OK: Memory usage is normal")
            print("   - System is ready for processing")
    
    def monitor_continuously(self, interval=30):
        """Monitor memory continuously."""
        print(f"🔍 Starting continuous memory monitoring (checking every {interval}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                self.print_status()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Memory monitoring for playlist generator')
    parser.add_argument('--continuous', '-c', action='store_true',
                        help='Monitor continuously')
    parser.add_argument('--interval', '-i', type=int, default=30,
                        help='Monitoring interval in seconds (default: 30)')
    parser.add_argument('--critical', type=int, default=85,
                        help='Critical memory threshold percentage (default: 85)')
    parser.add_argument('--warning', type=int, default=75,
                        help='Warning memory threshold percentage (default: 75)')
    
    args = parser.parse_args()
    
    monitor = MemoryMonitor(args.critical, args.warning)
    
    if args.continuous:
        monitor.monitor_continuously(args.interval)
    else:
        monitor.print_status()

if __name__ == "__main__":
    main() 