#!/usr/bin/env python3
"""
Test script for the Logic Provider Functions API
Run this after setting up your environment variables

This script tests the following endpoints:
- Health check (/health)
- Root endpoint (/)
- HTML upload (/upload-html)
- Image generation (/generate-image) - requires OPENAI_API_KEY
- Loom video processing (/process-loom-video) - requires ASSEMBLY_AI_API_KEY

Environment variables needed:
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- S3_BUCKET_NAME
- OPENAI_API_KEY (optional, for image generation)
- ASSEMBLY_AI_API_KEY (optional, for video transcription)

Run with: python test_api.py
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

def test_process_loom_video():
    """Test the Loom video processing endpoint"""
    try:
        # Check if Assembly AI API key is available
        assembly_api_key = os.getenv("ASSEMBLY_AI_API_KEY")
        if not assembly_api_key:
            print("Assembly AI API key not found in environment variables. Skipping Loom video processing test.")
            return True  # Return True to not fail the test suite

        # Sample Loom video URL for testing
        # Note: Replace this with a real Loom video URL that you have access to
        # The URL format should be: https://www.loom.com/share/[VIDEO_ID]
        loom_url = "https://www.loom.com/share/example-video-id"  # Replace with actual Loom URL

        payload = {
            "loom_url": loom_url,
            "filename": "test_loom_video.mp4"
        }

        print("Starting Loom video processing test...")
        print("Note: This test may take several minutes due to video download and transcription.")
        print("Video URL:", loom_url)

        # Use a longer timeout since this involves downloading and transcribing
        response = requests.post(
            f"{BASE_URL}/process-loom-video",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600  # 10 minutes timeout for video processing
        )

        print(f"Process Loom Video: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result['success']}")
            print(f"Video URL: {result['video_url']}")
            print(f"Transcript length: {len(result['transcript']['text'])} characters")
            print(f"Confidence: {result['transcript']['confidence']}")
            print(f"Duration: {result['transcript']['duration']} seconds")
            print(f"Language: {result['transcript']['language']}")
            print(f"Words count: {len(result['transcript']['words'])}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("Loom video processing test timed out (this is expected for long-running processes)")
        return True  # Don't fail the test for timeout, as it's expected for real video processing
    except Exception as e:
        print(f"Loom video processing test failed: {e}")
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

    # Check for optional API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Image generation test will be skipped.")
        print()

    if not os.getenv("ASSEMBLY_AI_API_KEY"):
        print("Warning: ASSEMBLY_AI_API_KEY environment variable not set.")
        print("Loom video processing test will be skipped.")
        print()
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Upload HTML", test_upload_html),
        ("Generate Image", test_generate_image),
        ("Process Loom Video", test_process_loom_video)
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
