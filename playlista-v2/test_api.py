#!/usr/bin/env python3
"""
Simple test script to verify Playlista v2 API functionality
"""

import requests
import json
import time

def test_api_endpoint(url, description):
    """Test a single API endpoint"""
    try:
        print(f"Testing {description}...")
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {description} - SUCCESS")
            print(f"   Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ {description} - FAILED (Status: {response.status_code})")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    """Run comprehensive API tests"""
    print("🧪 Testing Playlista v2 API Endpoints")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    tests = [
        (f"{base_url}/api/health", "Health Check"),
        (f"{base_url}/api/library/stats", "Library Statistics"),
        (f"{base_url}/api/library/tracks?limit=5", "Track Listing"),
        (f"{base_url}/api/analysis/jobs", "Analysis Jobs"),
    ]
    
    passed = 0
    total = len(tests)
    
    for url, description in tests:
        if test_api_endpoint(url, description):
            passed += 1
        print()
        time.sleep(1)
    
    print("=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API endpoints are working!")
        print("\n🌐 You can now access:")
        print(f"   Frontend: http://localhost:3000")
        print(f"   Backend API: http://localhost:8000/api")
        print(f"   API Docs: http://localhost:8000/docs")
    else:
        print("⚠️  Some endpoints are not working properly")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
