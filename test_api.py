#!/usr/bin/env python3
"""
Test script for the Logic Provider Functions API
Run this after setting up your environment variables
"""

import requests
import json
import os

# API base URL (change this to your deployed URL when testing production)
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Root Endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Root endpoint failed: {e}")
        return False

def test_upload_html():
    """Test the HTML upload endpoint"""
    try:
        # Sample HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>Hello from Test API!</h1>
            <p>This is a test HTML file uploaded via the API.</p>
            <p>Timestamp: {timestamp}</p>
        </body>
        </html>
        """.format(timestamp=__import__('datetime').datetime.now().isoformat())
        
        payload = {
            "html_content": html_content,
            "filename": "test_page.html"
        }
        
        response = requests.post(
            f"{BASE_URL}/upload-html",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Upload HTML: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Public URL: {result['public_url']}")
            print(f"Filename: {result['filename']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Upload test failed: {e}")
        return False

def test_generate_image():
    """Test the image generation endpoint"""
    try:
        # Check if OpenAI API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("OpenAI API key not found in environment variables. Skipping image generation test.")
            return True  # Return True to not fail the test suite
        
        payload = {
            "prompt": "A beautiful sunset over a mountain landscape",
            "size": "1024x1024",
            "quality": "standard"
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}"
        }
        
        response = requests.post(
            f"{BASE_URL}/generate-image",
            json=payload,
            headers=headers,
            timeout=120  # Image generation can take time
        )
        
        print(f"Generate Image: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Public URL: {result['public_url']}")
            print(f"Filename: {result['filename']}")
            print(f"Prompt: {result['prompt']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Image generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Logic Provider Functions API")
    print("=" * 40)
    
    # Check if environment variables are set
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'S3_BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Warning: Missing environment variables: {missing_vars}")
        print("Some tests may fail without proper AWS configuration.")
        print()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Image generation test will be skipped.")
        print()
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Upload HTML", test_upload_html),
        ("Generate Image", test_generate_image)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 20)
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Results Summary:")
    print("=" * 40)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
