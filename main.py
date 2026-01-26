from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import boto3
import uuid
from datetime import datetime
import os
from typing import Optional
import json
import logging
from dotenv import load_dotenv
import requests
import base64
import io
import tempfile
import time
import re
from urllib.parse import urlparse, parse_qs, unquote
from mutagen.mp3 import MP3
from io import BytesIO
import loomdl
import assemblyai as aai

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Pydantic model for request body
class HTMLUploadRequest(BaseModel):
    html_content: str
    filename: Optional[str] = None
    content_type: str = "text/html"

class ImageGenerationRequest(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"
    quality: Optional[str] = "auto"

class LoomVideoRequest(BaseModel):
    loom_url: str
    filename: Optional[str] = None

class LinkedInVideoRequest(BaseModel):
    video_url: str  # URL of the video to upload
    caption: str
    visibility: Optional[str] = "PUBLIC"  # PUBLIC, CONNECTIONS, or LOGGED_IN
    filename: Optional[str] = None

class HTMLToImageRequest(BaseModel):
    html_content: str
    width: Optional[int] = 1080  # Default Instagram post width
    height: Optional[int] = 1080  # Default Instagram post height (square)
    filename: Optional[str] = None
    quality: Optional[int] = 95  # PNG quality (0-100)

class UrlToGoogleDriveRequest(BaseModel):
    file_url: str  # Publicly accessible URL to download from
    folder_id: str  # Google Drive folder ID to upload to
    filename: Optional[str] = None  # Optional filename for the uploaded file

class ElevenLabsSpeechRequest(BaseModel):
    """
    Request model for Eleven Labs text-to-speech generation.
    Based on: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
    """
    voice_id: str  # Eleven Labs voice ID to use for speech generation
    text: str  # Text to convert to speech
    model_id: Optional[str] = "eleven_multilingual_v2"  # Model ID (default: eleven_multilingual_v2)
    filename: Optional[str] = None  # Optional filename for the uploaded audio
    # Voice settings (all optional - uses Eleven Labs defaults if not provided)
    stability: Optional[float] = None  # 0.0-1.0, default 0.5. Lower = more emotional range, higher = more stable/monotonous
    similarity_boost: Optional[float] = None  # 0.0-1.0, default 0.75. How closely AI adheres to original voice
    style: Optional[float] = None  # 0.0-1.0, default 0. Style exaggeration (higher = more computational resources)
    use_speaker_boost: Optional[bool] = None  # default true. Boosts similarity to original speaker (increases latency)
    speed: Optional[float] = None  # default 1.0. Speech speed (<1.0 slower, >1.0 faster)

class HeyGenAvatarIVRequest(BaseModel):
    """
    Request model for HeyGen Avatar IV video generation.
    Based on: https://docs.heygen.com/reference/create-avatar-iv-video
    
    Note: Instead of image_key, this endpoint accepts image_url which will be 
    automatically uploaded to HeyGen using their Upload Asset API.
    
    You can either:
    1. Use TTS (text-to-speech): Provide script and voice_id
    2. Use pre-recorded audio: Provide audio_url or audio_asset_id (script and voice_id not required)
    """
    image_url: str  # Publicly accessible URL of the image to use for avatar (will be uploaded to HeyGen)
    script: Optional[str] = None  # The text that the avatar will speak (required for TTS, optional if using audio_url)
    voice_id: Optional[str] = None  # The voice ID to use for speech generation (required for TTS, optional if using audio_url)
    video_title: Optional[str] = None  # Optional title for the video
    video_orientation: Optional[str] = None  # Video orientation (e.g., "landscape", "portrait", "square")
    fit: Optional[str] = None  # How the image fits in the video frame: "cover" or "contain"
    custom_motion_prompt: Optional[str] = None  # Custom motion prompt for avatar movements
    enhance_custom_motion_prompt: Optional[bool] = None  # Whether to enhance the custom motion prompt with AI
    audio_url: Optional[str] = None  # URL of an audio file to use instead of TTS
    audio_asset_id: Optional[str] = None  # Asset ID of a previously uploaded audio file

class GoogleDriveToS3Request(BaseModel):
    """
    Request model for downloading a file from Google Drive and uploading to S3.
    The Google Drive file must be publicly accessible (shared with "Anyone with the link").
    
    Accepts various Google Drive URL formats or just the file ID:
    - File ID only: "1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H"
    - Share link: "https://drive.google.com/file/d/1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H/view"
    - Download link: "https://drive.google.com/uc?id=1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H&export=download"
    - Open link: "https://drive.google.com/open?id=1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H"
    """
    file_id: str  # Google Drive file ID or full URL (ID will be extracted automatically)
    folder_path: str  # S3 folder path (e.g., "videos/", "images/uploads/")
    filename: Optional[str] = None  # Optional filename. If not provided, will try to get from Google Drive

class TranscribeAudioRequest(BaseModel):
    audio_url: str  # Publicly accessible URL to the audio file

app = FastAPI(title="Logic Provider Functions", version="1.0.0")

# AWS S3 configuration
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Assembly AI configuration
ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")

# Initialize S3 client
s3_client = None
try:
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        logger.info("S3 client initialized successfully")
    else:
        logger.warning("AWS credentials not fully configured - S3 operations will fail")
except Exception as e:
    logger.error(f"Failed to initialize S3 client: {e}")
    s3_client = None

# Initialize Assembly AI client
assembly_client = None
try:
    if ASSEMBLY_AI_API_KEY:
        aai.settings.api_key = ASSEMBLY_AI_API_KEY
        logger.info("Assembly AI client initialized successfully")
    else:
        logger.warning("Assembly AI API key not configured - transcription operations will fail")
except Exception as e:
    logger.error(f"Failed to initialize Assembly AI client: {e}")
    assembly_client = None

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Logic Provider Functions API", 
        "status": "running",
        "endpoints": ["/upload-html", "/generate-image", "/html-to-image", "/process-loom-video", "/upload-linkedin-video", "/url-to-google-drive", "/eleven-labs-speech", "/heygen-avatar-iv", "/google-drive-to-s3", "/telegram-file/bot{BotToken}/{file_path}", "/transcribe-audio", "/health"],
        "aws_configured": bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-html")
async def upload_html(request: HTMLUploadRequest):
    """
    Upload HTML content to AWS S3 bucket and return public URL
    
    Args:
        request: HTMLUploadRequest containing html_content and optional filename
    
    Returns:
        JSON response with upload status and public URL
    """
    logger.info(f"Upload HTML request received - Filename: {request.filename}, Content length: {len(request.html_content)}")
    
    try:
        # Validate required environment variables
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500, 
                detail="AWS credentials or bucket name not configured"
            )
        
        # Check if S3 client is available
        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )
        
        logger.info(f"Request details: {request}")
        
        # Extract values from request
        html_content = request.html_content
        filename = request.filename
        content_type = request.content_type
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"html/html_{timestamp}_{unique_id}.html"
            logger.info(f"Generated filename: {filename}")
        else:
            # Ensure filename has html/ prefix and .html extension
            if not filename.startswith("html/"):
                filename = f"html/{filename}"
            if not filename.endswith('.html'):
                filename += '.html'
            logger.info(f"Adjusted filename: {filename}")
        
        logger.info(f"Attempting to upload to S3 - Bucket: {S3_BUCKET_NAME}, Key: {filename}")
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=html_content.encode('utf-8'),
            ContentType=content_type,
            ACL='public-read'  # Make the object publicly readable
        )
        
        logger.info(f"Successfully uploaded to S3: {filename}")
        
        # Generate public URL
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        logger.info(f"Generated public URL: {public_url}")
        
        response_data = {
            "success": True,
            "message": "HTML content uploaded successfully",
            "filename": filename,
            "public_url": public_url,
            "bucket": S3_BUCKET_NAME,
            "region": AWS_REGION,
            "key": filename,
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Upload successful - returning response: {response_data}")
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload HTML content: {str(e)}"
        )

@app.post("/generate-image")
async def generate_image(
    request: ImageGenerationRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Generate an image using OpenAI's DALL-E API and save it to S3
    
    Args:
        request: ImageGenerationRequest containing prompt and optional size/quality
        authorization: Bearer token for OpenAI API access
    
    Returns:
        JSON response with generation status and S3 URL
    """
    logger.info(f"Image generation request received - Prompt length: {len(request.prompt)}")
    
    try:
        # Validate authorization header
        if not authorization or not authorization.lower().startswith("bearer "):
            logger.error("Missing or invalid authorization header")
            raise HTTPException(
                status_code=401,
                detail="Authorization header with Bearer token is required"
            )
        
        # Extract API key from authorization header (case insensitive)
        api_key = authorization.split(" ", 1)[1].strip()
        
        # Validate required AWS environment variables
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500,
                detail="AWS credentials or bucket name not configured"
            )
        
        # Check if S3 client is available
        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )
        
        # Prepare OpenAI API request
        openai_url = "https://api.openai.com/v1/images/generations"
        openai_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        openai_payload = {
            "model": "gpt-image-1",
            "prompt": request.prompt,
            "size": request.size,
            "quality": request.quality
        }
        
        logger.info(f"Making request to OpenAI API with payload: {openai_payload}")
        
        # Make request to OpenAI API
        response = requests.post(
            openai_url,
            headers=openai_headers,
            json=openai_payload,
            timeout=60  # 60 second timeout for image generation
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {response.text}"
            )
        
        response_data = response.json()
        logger.info(f"OpenAI API response received successfully")
        
        # Extract base64 image data
        if not response_data.get("data") or len(response_data["data"]) == 0:
            logger.error("No image data returned from OpenAI API")
            raise HTTPException(
                status_code=500,
                detail="No image data returned from OpenAI API"
            )
        
        b64_json = response_data["data"][0]["b64_json"]
        
        # Decode base64 image
        image_data = base64.b64decode(b64_json)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"ai-generated-images/image_{timestamp}_{unique_id}.png"
        
        logger.info(f"Attempting to upload image to S3 - Bucket: {S3_BUCKET_NAME}, Key: {filename}")
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=image_data,
            ContentType="image/png",
            ACL='public-read'  # Make the object publicly readable
        )
        
        logger.info(f"Successfully uploaded image to S3: {filename}")
        
        # Generate public URL
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        logger.info(f"Generated public URL: {public_url}")
        
        response_result = {
            "success": True,
            "message": "Image generated and uploaded successfully",
            "public_url": public_url,
            "prompt": request.prompt,
            "size": request.size,
            "quality": request.quality,
            "generated_at": datetime.now().isoformat(),
        }
        
        logger.info(f"Image generation successful - returning response")
        return JSONResponse(
            status_code=200,
            content=response_result
        )
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during OpenAI API call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to OpenAI API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate image: {str(e)}"
        )

@app.post("/process-loom-video")
async def process_loom_video(request: LoomVideoRequest):
    """
    Download a Loom video, upload it to S3, transcribe it with Assembly AI, and return both the video URL and transcript.

    Args:
        request: LoomVideoRequest containing loom_url and optional filename

    Returns:
        JSON response with video URL and transcription result
    """
    logger.info(f"Loom video processing request received - URL: {request.loom_url}")

    try:
        # Validate required configurations
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500,
                detail="AWS credentials or bucket name not configured"
            )

        if not ASSEMBLY_AI_API_KEY:
            logger.error("Missing Assembly AI API key")
            raise HTTPException(
                status_code=500,
                detail="Assembly AI API key not configured"
            )

        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )

        # Generate filename if not provided
        filename = request.filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"videos/loom_video_{timestamp}_{unique_id}.mp4"
            logger.info(f"Generated filename: {filename}")
        else:
            # Ensure it has the videos/ prefix and .mp4 extension
            if not filename.startswith("videos/"):
                filename = f"videos/{filename}"
            if not filename.endswith('.mp4'):
                filename += '.mp4'
            logger.info(f"Using provided filename: {filename}")

        # Step 1: Download video using loom-dl
        logger.info(f"Downloading video from: {request.loom_url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_video_path = temp_file.name

        try:
            # Download video using loom-dl functions
            video_id = loomdl.extract_id(request.loom_url)
            download_url = loomdl.fetch_loom_download_url(video_id)
            loomdl.download_loom_video(download_url, temp_video_path)
            logger.info(f"Video downloaded successfully to: {temp_video_path}")

            # Step 2: Upload to S3
            logger.info(f"Uploading video to S3 - Bucket: {S3_BUCKET_NAME}, Key: {filename}")
            with open(temp_video_path, 'rb') as video_file:
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=filename,
                    Body=video_file,
                    ContentType="video/mp4",
                    ACL='public-read'
                )
            # Generate public URL
            video_public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
            logger.info(f"Video uploaded to S3 successfully: {video_public_url}")

            # Step 3: Send to Assembly AI for transcription
            logger.info(f"Sending video to Assembly AI for transcription: {video_public_url}")
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(video_public_url)

            # Step 4: Poll for transcription completion
            logger.info("Polling for transcription completion...")
            max_polls = 60  # Maximum 5 minutes (60 * 5 seconds)
            poll_count = 0

            while poll_count < max_polls:
                if transcript.status == aai.TranscriptStatus.completed:
                    logger.info("Transcription completed successfully")
                    break
                elif transcript.status == aai.TranscriptStatus.error:
                    logger.error(f"Transcription failed: {transcript.error}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Transcription failed: {transcript.error}"
                    )

                logger.info(f"Transcription status: {transcript.status}, waiting...")
                time.sleep(5)  # Wait 5 seconds before polling again
                poll_count += 1

            if poll_count >= max_polls:
                logger.error("Transcription timed out")
                raise HTTPException(
                    status_code=500,
                    detail="Transcription timed out after 5 minutes"
                )

            # Clean up temporary file
            os.unlink(temp_video_path)

            # Return response
            response_data = {
                "success": True,
                "message": "Video processed and transcribed successfully",
                "video_url": video_public_url,
                "transcript": transcript.text,
                # "transcript": {
                #     "text": transcript.text,
                #     "confidence": transcript.confidence,
                #     "duration": transcript.audio_duration,
                #     "words": [{"text": word.text, "start": word.start, "end": word.end, "confidence": word.confidence}
                #              for word in transcript.words] if transcript.words else []
                # },
                "filename": filename,
                "processed_at": datetime.now().isoformat()
            }

            logger.info("Loom video processing completed successfully")
            return JSONResponse(status_code=200, content=response_data)

        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            raise

    except Exception as e:
        logger.error(f"Error during loom video processing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process loom video: {str(e)}"
        )

@app.post("/html-to-image")
async def html_to_image(request: HTMLToImageRequest):
    """
    Convert HTML content to a PNG image optimized for Instagram posting.
    Uses Playwright to render HTML in a headless browser and capture as PNG.
    
    Instagram Recommended Dimensions:
    - Square Post: 1080x1080px (default)
    - Portrait Post: 1080x1350px
    - Landscape Post: 1080x566px
    - Story: 1080x1920px
    
    Args:
        request: HTMLToImageRequest containing html_content and optional dimensions/filename
    
    Returns:
        JSON response with the public S3 URL of the generated PNG image
    """
    logger.info(f"HTML-to-Image request received - HTML length: {len(request.html_content)}, Dimensions: {request.width}x{request.height}")
    
    try:
        # Validate required AWS environment variables
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500,
                detail="AWS credentials or bucket name not configured"
            )
        
        # Check if S3 client is available
        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )
        
        # Validate dimensions for Instagram
        if request.width < 320 or request.width > 1080:
            logger.warning(f"Width {request.width}px is outside Instagram's recommended range (320-1080px)")
        
        if request.height < 566 or request.height > 1920:
            logger.warning(f"Height {request.height}px is outside Instagram's recommended range (566-1920px)")
        
        # Import Playwright
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            logger.error("Playwright not installed")
            raise HTTPException(
                status_code=500,
                detail="Playwright library not installed. Please run: pip install playwright && playwright install chromium"
            )
        
        # Create temporary file for the screenshot
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_image_path = temp_file.name
        
        try:
            logger.info("Launching Playwright browser...")
            
            # Launch Playwright and render HTML
            with sync_playwright() as p:
                # Launch browser - use chromium for best compatibility
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-accelerated-2d-canvas',
                        '--no-first-run',
                        '--no-zygote',
                        '--disable-gpu'
                    ]
                )
                
                logger.info("Browser launched successfully")
                
                # Create a new page with specified viewport
                page = browser.new_page(
                    viewport={'width': request.width, 'height': request.height},
                    device_scale_factor=2  # Retina display for higher quality
                )
                
                logger.info(f"Page created with viewport: {request.width}x{request.height}")
                
                # Set the HTML content
                page.set_content(request.html_content, wait_until='networkidle')
                
                logger.info("HTML content loaded successfully")
                
                # Take screenshot
                page.screenshot(
                    path=temp_image_path,
                    type='png',
                    full_page=False,  # Only capture viewport
                    omit_background=False  # Include background
                )
                
                logger.info(f"Screenshot captured: {temp_image_path}")
                
                # Close browser
                browser.close()
                logger.info("Browser closed")
            
            # Generate filename if not provided
            filename = request.filename
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"instagram-images/ig_post_{timestamp}_{unique_id}.png"
            else:
                # Ensure filename has instagram-images/ prefix and .png extension
                if not filename.startswith("instagram-images/"):
                    filename = f"instagram-images/{filename}"
                if not filename.endswith('.png'):
                    filename += '.png'
            
            logger.info(f"Uploading image to S3 - Bucket: {S3_BUCKET_NAME}, Key: {filename}")
            
            # Upload to S3
            with open(temp_image_path, 'rb') as image_file:
                s3_client.put_object(
                    Bucket=S3_BUCKET_NAME,
                    Key=filename,
                    Body=image_file,
                    ContentType="image/png",
                    ACL='public-read',
                    Metadata={
                        'width': str(request.width),
                        'height': str(request.height),
                        'generated-by': 'html-to-image-api'
                    }
                )
            
            logger.info(f"Image uploaded to S3 successfully: {filename}")
            
            # Generate public URL
            public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
            logger.info(f"Generated public URL: {public_url}")
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            # Return response
            response_data = {
                "success": True,
                "message": "HTML converted to image successfully",
                "public_url": public_url,
                "filename": filename,
                "dimensions": {
                    "width": request.width,
                    "height": request.height
                },
                "instagram_optimized": True,
                "format": "PNG",
                "generated_at": datetime.now().isoformat()
            }
            
            logger.info("HTML-to-Image conversion completed successfully")
            return JSONResponse(status_code=200, content=response_data)
            
        except Exception as e:
            # Clean up temporary file if it exists
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise
            
    except Exception as e:
        logger.error(f"Error during HTML-to-Image conversion: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert HTML to image: {str(e)}"
        )

@app.post("/upload-linkedin-video")
async def upload_linkedin_video(
    request: LinkedInVideoRequest,
    authorization: str = Header(...)
):
    """
    Upload a video to LinkedIn and create a UGC post with caption using LinkedIn v2 API.

    Video Requirements:
    - Format: MP4
    - Duration: 3 seconds to 30 minutes
    - Size: 75 KB to 500 MB

    The LinkedIn access token must have the following permissions:
    - r_liteprofile: Read basic profile information
    - w_member_social: Write posts and content

    Uses LinkedIn v2 API endpoints in a 3-step process:
    - POST /v2/assets?action=registerUpload (initialize upload)
    - PUT to upload URL (upload video file)
    - POST /v2/ugcPosts (create UGC post)

    Args:
        request: LinkedInVideoRequest containing video_url, caption, and optional visibility/filename
        authorization: Bearer token for LinkedIn API access (should be "Bearer <token>")

    Returns:
        JSON response with upload status and post details
    """
    logger.info(f"LinkedIn video upload request received - Video URL: {request.video_url}")

    try:
        # Validate authorization header
        if not authorization or not authorization.lower().startswith("bearer "):
            logger.error("Missing or invalid authorization header")
            raise HTTPException(
                status_code=401,
                detail="Authorization header with Bearer token is required"
            )

        # Extract access token
        access_token = authorization.split(" ", 1)[1].strip()

        # Validate visibility setting
        valid_visibilities = ["PUBLIC", "CONNECTIONS", "LOGGED_IN"]
        if request.visibility.upper() not in valid_visibilities:
            logger.error(f"Invalid visibility setting: {request.visibility}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid visibility. Must be one of: {', '.join(valid_visibilities)}"
            )

        # Step 1: Download the video from the provided URL
        logger.info(f"Downloading video from: {request.video_url}")
        video_response = requests.get(request.video_url, timeout=300)  # 5 minute timeout for large videos
        video_response.raise_for_status()

        video_content = video_response.content
        content_length = len(video_content)

        # Validate video specifications per LinkedIn requirements
        min_size_kb = 75 * 1024  # 75 KB
        max_size_mb = 500 * 1024 * 1024  # 500 MB

        if content_length < min_size_kb:
            logger.error(f"Video too small: {content_length} bytes. Minimum: {min_size_kb} bytes (75 KB)")
            raise HTTPException(
                status_code=400,
                detail="Video file is too small. LinkedIn requires minimum 75 KB"
            )

        if content_length > max_size_mb:
            logger.error(f"Video too large: {content_length} bytes. Maximum: {max_size_mb} bytes (500 MB)")
            raise HTTPException(
                status_code=400,
                detail="Video file is too large. LinkedIn allows maximum 500 MB"
            )

        logger.info(f"Video downloaded successfully. Size: {content_length} bytes ({content_length / (1024*1024):.2f} MB)")

        # Generate filename if not provided
        filename = request.filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"linkedin_video_{timestamp}_{unique_id}.mp4"
        elif not filename.endswith('.mp4'):
            filename += '.mp4'

        # Step 2: Get user's member ID from LinkedIn
        logger.info("Getting user member ID from LinkedIn API")

        linkedin_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": "202409"  # Required for LinkedIn REST API
        }

        # Get the authenticated user's info to get their member ID
        profile_url = "https://api.linkedin.com/v2/userinfo"
        profile_response = requests.get(
            profile_url,
            headers=linkedin_headers,
            timeout=30
        )

        if profile_response.status_code != 200:
            logger.error(f"Failed to get LinkedIn userinfo: {profile_response.status_code} - {profile_response.text}")
            raise HTTPException(
                status_code=profile_response.status_code,
                detail=f"Failed to get LinkedIn user information: {profile_response.text}"
            )

        profile_data = profile_response.json()
        member_id = profile_data.get("sub")

        if not member_id:
            logger.error(f"Could not extract member ID from userinfo response: {profile_data}")
            raise HTTPException(
                status_code=500,
                detail="Could not extract member ID from LinkedIn userinfo"
            )

        logger.info(f"Retrieved member ID: {member_id}")

        # Step 3: Initialize the Upload with LinkedIn v2 API
        logger.info("Initializing video upload with LinkedIn v2 API")

        register_payload = {
            "registerUploadRequest": {
                "recipes": [
                    "urn:li:digitalmediaRecipe:feedshare-video"
                ],
                "owner": f"urn:li:person:{member_id}",
                "serviceRelationships": [
                    {
                        "relationshipType": "OWNER",
                        "identifier": "urn:li:userGeneratedContent"
                    }
                ]
            }
        }

        register_url = "https://api.linkedin.com/v2/assets?action=registerUpload"
        register_response = requests.post(
            register_url,
            headers=linkedin_headers,
            json=register_payload,
            timeout=30
        )

        if register_response.status_code != 200:
            logger.error(f"LinkedIn video registration failed: {register_response.status_code} - {register_response.text}")
            raise HTTPException(
                status_code=register_response.status_code,
                detail=f"Failed to register video upload: {register_response.text}"
            )

        register_data = register_response.json()

        # Extract asset URN and upload URL from v2 API response
        asset_urn = register_data.get("value", {}).get("asset")

        upload_mechanism = register_data.get("value", {}).get("uploadMechanism", {})
        upload_url = upload_mechanism.get("com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest", {}).get("uploadUrl")

        if not asset_urn or not upload_url:
            logger.error(f"Invalid response from LinkedIn v2 API registration: {register_data}")
            raise HTTPException(
                status_code=500,
                detail="Invalid response from LinkedIn v2 API during registration"
            )

        logger.info(f"Video registered successfully. Asset URN: {asset_urn}")

        # Step 4: Upload the Video File
        logger.info(f"Uploading video to LinkedIn upload URL: {upload_url}")

        # Note: Do NOT include Authorization header in this PUT request
        upload_headers = {
            "Content-Type": "video/mp4"
        }

        upload_response = requests.put(
            upload_url,
            headers=upload_headers,
            data=video_content,
            timeout=600  # 10 minute timeout for upload
        )

        if upload_response.status_code not in [200, 201]:
            logger.error(f"LinkedIn video upload failed: {upload_response.status_code} - {upload_response.text}")
            raise HTTPException(
                status_code=upload_response.status_code,
                detail=f"Failed to upload video: {upload_response.text}"
            )

        logger.info("Video uploaded successfully to LinkedIn")

        # Step 5: Create the Post (UGC Post)
        logger.info("Creating LinkedIn UGC post with video")

        post_payload = {
            "author": f"urn:li:person:{member_id}",
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {
                        "text": request.caption
                    },
                    "shareMediaCategory": "VIDEO",
                    "media": [
                        {
                            "status": "READY",
                            "description": {
                                "text": "Video uploaded via API"
                            },
                            "media": asset_urn
                        }
                    ]
                }
            },
            "visibility": {
                "com.linkedin.ugc.MemberNetworkVisibility": request.visibility.upper()
            }
        }

        post_url = "https://api.linkedin.com/v2/ugcPosts"
        post_response = requests.post(
            post_url,
            headers=linkedin_headers,
            json=post_payload,
            timeout=30
        )

        if post_response.status_code not in [200, 201]:
            logger.error(f"LinkedIn UGC post creation failed: {post_response.status_code} - {post_response.text}")
            raise HTTPException(
                status_code=post_response.status_code,
                detail=f"Failed to create UGC post: {post_response.text}"
            )

        post_data = post_response.json()
        post_urn = post_data.get("id")

        if not post_urn:
            logger.warning(f"No post ID in UGC Posts API response: {post_data}")
            post_urn = post_data.get("urn") or f"urn:li:share:{post_data.get('id', 'unknown')}"

        logger.info(f"LinkedIn UGC post created successfully. Post URN: {post_urn}")

        # Return success response
        response_data = {
            "success": True,
            "message": "Video uploaded and posted to LinkedIn successfully",
            "asset_urn": asset_urn,
            "post_urn": post_urn,
            "caption": request.caption,
            "visibility": request.visibility,
            "filename": filename,
            "uploaded_at": datetime.now().isoformat()
        }

        logger.info("LinkedIn video upload completed successfully")
        return JSONResponse(status_code=200, content=response_data)

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during LinkedIn API call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to LinkedIn API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during LinkedIn video upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload video to LinkedIn: {str(e)}"
        )

@app.post("/url-to-google-drive")
async def url_to_google_drive(
    request: UrlToGoogleDriveRequest,
    authorization: str = Header(...)
):
    """
    Download a file from a publicly accessible URL and upload it to Google Drive.
    
    This endpoint acts as a middleman that:
    1. Downloads the file from the provided URL
    2. Uploads it to the specified Google Drive folder using the provided credentials
    
    Args:
        request: UrlToGoogleDriveRequest containing file_url, folder_id, and optional filename
        authorization: Bearer token for Google Drive API access (should be "Bearer <access_token>")
    
    Returns:
        JSON response with upload status and Google Drive file details
    """
    logger.info(f"URL to Google Drive request received - URL: {request.file_url}, Folder ID: {request.folder_id}")
    
    try:
        # Validate authorization header
        if not authorization or not authorization.lower().startswith("bearer "):
            logger.error("Missing or invalid authorization header")
            raise HTTPException(
                status_code=401,
                detail="Authorization header with Bearer token is required. Format: 'Bearer <google_drive_access_token>'"
            )
        
        # Extract access token
        access_token = authorization.split(" ", 1)[1].strip()
        
        # Step 1: Download the file from the provided URL
        logger.info(f"Downloading file from: {request.file_url}")
        
        try:
            download_response = requests.get(
                request.file_url,
                timeout=300,  # 5 minute timeout for large files
                stream=True
            )
            download_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from URL: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download file from URL: {str(e)}"
            )
        
        file_content = download_response.content
        content_length = len(file_content)
        
        # Try to determine content type from response headers
        content_type = download_response.headers.get('Content-Type', 'application/octet-stream')
        # Clean up content type (remove charset and other parameters)
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()
        
        logger.info(f"File downloaded successfully. Size: {content_length} bytes ({content_length / (1024*1024):.2f} MB), Content-Type: {content_type}")
        
        # Step 2: Determine filename
        filename = request.filename
        if not filename:
            # Try to extract filename from URL or Content-Disposition header
            content_disposition = download_response.headers.get('Content-Disposition', '')
            if 'filename=' in content_disposition:
                match = re.search(r'filename[*]?=["\']?([^"\';\s]+)["\']?', content_disposition)
                if match:
                    filename = match.group(1)
            
            if not filename:
                # Extract from URL path
                parsed_url = urlparse(request.file_url)
                path_filename = parsed_url.path.split('/')[-1]
                if path_filename:
                    filename = unquote(path_filename)
                else:
                    # Generate a default filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    # Try to determine extension from content type
                    extension_map = {
                        'image/jpeg': '.jpg',
                        'image/png': '.png',
                        'image/gif': '.gif',
                        'image/webp': '.webp',
                        'video/mp4': '.mp4',
                        'video/webm': '.webm',
                        'application/pdf': '.pdf',
                        'text/plain': '.txt',
                        'text/html': '.html',
                        'application/json': '.json',
                    }
                    extension = extension_map.get(content_type, '')
                    filename = f"file_{timestamp}_{unique_id}{extension}"
        
        logger.info(f"Using filename: {filename}")
        
        # Step 3: Upload to Google Drive using the REST API
        logger.info(f"Uploading file to Google Drive folder: {request.folder_id}")
        
        # Google Drive API v3 - multipart upload
        # First, create metadata
        metadata = {
            "name": filename,
            "parents": [request.folder_id]
        }
        
        # Use multipart upload for simplicity
        import io
        from email.mime.multipart import MIMEMultipart
        from email.mime.base import MIMEBase
        from email.mime.application import MIMEApplication
        
        # Create the multipart request manually
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex[:16]}"
        
        # Build the multipart body
        body_parts = []
        
        # Part 1: Metadata (JSON)
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(b'Content-Type: application/json; charset=UTF-8')
        body_parts.append(b'')
        body_parts.append(json.dumps(metadata).encode())
        
        # Part 2: File content
        body_parts.append(f'--{boundary}'.encode())
        body_parts.append(f'Content-Type: {content_type}'.encode())
        body_parts.append(b'')
        body_parts.append(file_content)
        
        # Closing boundary
        body_parts.append(f'--{boundary}--'.encode())
        
        # Join with CRLF
        body = b'\r\n'.join(body_parts)
        
        # Make the upload request
        upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
        
        upload_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": f"multipart/related; boundary={boundary}"
        }
        
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            data=body,
            timeout=600  # 10 minute timeout for large uploads
        )
        
        if upload_response.status_code not in [200, 201]:
            logger.error(f"Google Drive upload failed: {upload_response.status_code} - {upload_response.text}")
            raise HTTPException(
                status_code=upload_response.status_code,
                detail=f"Failed to upload to Google Drive: {upload_response.text}"
            )
        
        drive_response = upload_response.json()
        file_id = drive_response.get("id")
        file_name = drive_response.get("name")
        
        logger.info(f"File uploaded to Google Drive successfully. File ID: {file_id}")
        
        # Generate Google Drive file URL
        drive_file_url = f"https://drive.google.com/file/d/{file_id}/view"
        
        # Return success response
        response_data = {
            "success": True,
            "message": "File downloaded and uploaded to Google Drive successfully",
            "file_id": file_id,
            "file_name": file_name,
            "drive_url": drive_file_url,
            "folder_id": request.folder_id,
            "source_url": request.file_url,
            "file_size_bytes": content_length,
            "content_type": content_type,
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info("URL to Google Drive transfer completed successfully")
        return JSONResponse(status_code=200, content=response_data)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during Google Drive API call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Google Drive API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during URL to Google Drive transfer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transfer file to Google Drive: {str(e)}"
        )

@app.post("/eleven-labs-speech")
async def eleven_labs_speech(
    request: ElevenLabsSpeechRequest,
    xi_api_key: str = Header(..., alias="xi-api-key")
):
    """
    Generate speech using Eleven Labs Text-to-Speech API and upload to S3.
    
    This endpoint:
    1. Takes the xi-api-key from request headers
    2. Calls Eleven Labs API to generate speech from text
    3. Uploads the returned audio file to S3 under the audio/ folder
    4. Returns the public URL of the audio file
    
    Args:
        request: ElevenLabsSpeechRequest containing voice_id, text, optional model_id and filename
        xi_api_key: Eleven Labs API key passed via xi-api-key header
    
    Returns:
        JSON response with the public S3 URL of the generated audio file
    """
    logger.info(f"Eleven Labs speech request received - Voice ID: {request.voice_id}, Text length: {len(request.text)}")
    
    try:
        # Validate required AWS environment variables
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500,
                detail="AWS credentials or bucket name not configured"
            )
        
        # Check if S3 client is available
        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )
        
        # Validate text is not empty
        if not request.text.strip():
            logger.error("Empty text provided")
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        # Prepare Eleven Labs API request
        eleven_labs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{request.voice_id}"
        eleven_labs_headers = {
            "xi-api-key": xi_api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg"
        }
        
        # Build voice_settings object with provided values or defaults
        voice_settings = {}
        
        # Stability: 0.0-1.0, default 0.5
        voice_settings["stability"] = request.stability if request.stability is not None else 0.5
        
        # Similarity boost: 0.0-1.0, default 0.75
        voice_settings["similarity_boost"] = request.similarity_boost if request.similarity_boost is not None else 0.75
        
        # Style: 0.0-1.0, default 0 (only add if provided since it increases latency)
        if request.style is not None:
            voice_settings["style"] = request.style
        
        # Use speaker boost: default true (only add if explicitly set)
        if request.use_speaker_boost is not None:
            voice_settings["use_speaker_boost"] = request.use_speaker_boost
        
        # Speed: default 1.0 (only add if provided)
        if request.speed is not None:
            voice_settings["speed"] = request.speed
        
        eleven_labs_payload = {
            "text": request.text,
            "model_id": request.model_id,
            "voice_settings": voice_settings
        }
        
        logger.info(f"Making request to Eleven Labs API for voice: {request.voice_id}, voice_settings: {voice_settings}")
        
        # Make request to Eleven Labs API
        response = requests.post(
            eleven_labs_url,
            headers=eleven_labs_headers,
            json=eleven_labs_payload,
            timeout=120  # 2 minute timeout for speech generation
        )
        
        if response.status_code != 200:
            logger.error(f"Eleven Labs API error: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Eleven Labs API error: {response.text}"
            )
        
        # Get the audio content
        audio_content = response.content
        content_length = len(audio_content)
        
        # Calculate audio duration using mutagen
        audio_duration_seconds = None
        try:
            audio_file = BytesIO(audio_content)
            audio = MP3(audio_file)
            audio_duration_seconds = round(audio.info.length, 2)  # Round to 2 decimal places
            logger.info(f"Audio duration: {audio_duration_seconds} seconds")
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {str(e)}")
        
        logger.info(f"Audio generated successfully. Size: {content_length} bytes ({content_length / 1024:.2f} KB)")
        
        # Generate filename if not provided
        filename = request.filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"audio/speech_{timestamp}_{unique_id}.mp3"
        else:
            # Ensure filename has audio/ prefix and .mp3 extension
            if not filename.startswith("audio/"):
                filename = f"audio/{filename}"
            if not filename.endswith('.mp3'):
                filename += '.mp3'
        
        logger.info(f"Uploading audio to S3 - Bucket: {S3_BUCKET_NAME}, Key: {filename}")
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            Body=audio_content,
            ContentType="audio/mpeg",
            ACL='public-read'
        )
        
        logger.info(f"Audio uploaded to S3 successfully: {filename}")
        
        # Generate public URL
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{filename}"
        logger.info(f"Generated public URL: {public_url}")
        
        # Return success response
        response_data = {
            "success": True,
            "message": "Speech generated and uploaded successfully",
            "public_url": public_url,
            "filename": filename,
            "voice_id": request.voice_id,
            "model_id": request.model_id,
            "voice_settings": voice_settings,
            "text_length": len(request.text),
            "audio_size_bytes": content_length,
            "audio_duration_seconds": audio_duration_seconds,
            "generated_at": datetime.now().isoformat()
        }
        
        logger.info("Eleven Labs speech generation completed successfully")
        return JSONResponse(status_code=200, content=response_data)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during Eleven Labs API call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to Eleven Labs API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error during speech generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate speech: {str(e)}"
        )

@app.post("/heygen-avatar-iv")
async def heygen_avatar_iv(
    request: HeyGenAvatarIVRequest,
    x_api_key: str = Header(..., alias="X-Api-Key")
):
    """
    Generate an Avatar IV video using HeyGen's API.
    
    This endpoint:
    1. Takes a publicly accessible image URL
    2. Uploads the image to HeyGen using their Upload Asset API
    3. Uses the uploaded image to generate an Avatar IV video
    4. Returns the response from HeyGen
    
    HeyGen API Documentation:
    - Upload Asset: https://docs.heygen.com/reference/upload-asset
    - Create Avatar IV Video: https://docs.heygen.com/reference/create-avatar-iv-video
    
    Args:
        request: HeyGenAvatarIVRequest containing:
            - image_url: Publicly accessible URL of the image (required)
            - script: Text for the avatar to speak (required)
            - voice_id: Voice ID to use (required)
            - video_title: Optional title for the video
            - video_orientation: Optional video orientation
            - fit: Optional fit mode ("cover" or "contain")
            - custom_motion_prompt: Optional custom motion prompt
            - enhance_custom_motion_prompt: Optional boolean to enhance motion prompt
            - audio_url: Optional URL of audio file to use instead of TTS
            - audio_asset_id: Optional asset ID of uploaded audio
        x_api_key: HeyGen API key passed via X-Api-Key header
    
    Returns:
        JSON response from HeyGen's Create Avatar IV Video API
    """
    logger.info(f"HeyGen Avatar IV request received - Image URL: {request.image_url}, Script: {'Yes' if request.script else 'No'}, Audio URL: {'Yes' if request.audio_url else 'No'}")
    
    try:
        # Step 1: Download the image from the provided URL
        logger.info(f"Downloading image from URL: {request.image_url}")
        
        try:
            # Use a proper User-Agent header and handle URL encoding
            download_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            image_response = requests.get(
                request.image_url,
                headers=download_headers,
                timeout=60,  # 1 minute timeout for image download
                stream=True
            )
            image_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download image from URL: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download image from URL: {str(e)}"
            )
        
        image_content = image_response.content
        content_length = len(image_content)
        
        if content_length == 0:
            logger.error("Downloaded image is empty")
            raise HTTPException(
                status_code=400,
                detail="Downloaded image is empty. Please check the image URL."
            )
        
        # Determine content type from response headers
        content_type = image_response.headers.get('Content-Type', 'image/png')
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()
        
        # Determine file extension based on content type
        extension_map = {
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/webp': '.webp',
        }
        extension = extension_map.get(content_type, '.png')
        
        logger.info(f"Image downloaded successfully. Size: {content_length} bytes, Content-Type: {content_type}")
        
        # Step 2: Upload the image to HeyGen as raw binary data
        logger.info("Uploading image to HeyGen...")
        
        upload_url = "https://upload.heygen.com/v1/asset"
        upload_headers = {
            "X-Api-Key": x_api_key,
            "Content-Type": content_type  # Set content type to actual image type (e.g., image/png)
        }
        
        # Upload as raw binary data (not multipart form-data)
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            data=image_content,  # Raw binary data
            timeout=120  # 2 minute timeout for upload
        )
        
        logger.info(f"HeyGen upload response status: {upload_response.status_code}")
        logger.info(f"HeyGen upload response body: {upload_response.text}")
        
        if upload_response.status_code != 200:
            logger.error(f"HeyGen asset upload failed: {upload_response.status_code} - {upload_response.text}")
            raise HTTPException(
                status_code=upload_response.status_code,
                detail=f"Failed to upload image to HeyGen: {upload_response.text}"
            )
        
        upload_data = upload_response.json()
        logger.info(f"HeyGen upload response parsed: {upload_data}")
        
        # Extract the image key from the response
        # HeyGen returns the asset_id in data.image_key or data.asset_id or data.url
        image_key = None
        if "data" in upload_data:
            image_key = upload_data["data"].get("image_key") or upload_data["data"].get("asset_id") or upload_data["data"].get("url")
        elif "image_key" in upload_data:
            image_key = upload_data["image_key"]
        elif "asset_id" in upload_data:
            image_key = upload_data["asset_id"]
        
        if not image_key:
            logger.error(f"Could not extract image_key from HeyGen upload response: {upload_data}")
            raise HTTPException(
                status_code=500,
                detail=f"Could not extract image_key from HeyGen upload response: {upload_data}"
            )
        
        logger.info(f"Image uploaded to HeyGen successfully. Image key: {image_key}")
        
        # Step 3: Generate the Avatar IV video
        logger.info("Generating Avatar IV video...")
        
        generate_url = "https://api.heygen.com/v2/video/av4/generate"
        generate_headers = {
            "X-Api-Key": x_api_key,
            "Content-Type": "application/json"
        }
        
        # Build the request payload with image_key (required)
        generate_payload = {
            "image_key": image_key
        }
        
        # Add TTS parameters if provided (script and voice_id)
        if request.script is not None:
            generate_payload["script"] = request.script
        if request.voice_id is not None:
            generate_payload["voice_id"] = request.voice_id
        
        # Add optional parameters if provided
        if request.video_title is not None:
            generate_payload["video_title"] = request.video_title
        if request.video_orientation is not None:
            generate_payload["video_orientation"] = request.video_orientation
        if request.fit is not None:
            generate_payload["fit"] = request.fit
        if request.custom_motion_prompt is not None:
            generate_payload["custom_motion_prompt"] = request.custom_motion_prompt
        if request.enhance_custom_motion_prompt is not None:
            generate_payload["enhance_custom_motion_prompt"] = request.enhance_custom_motion_prompt
        if request.audio_url is not None:
            generate_payload["audio_url"] = request.audio_url
        if request.audio_asset_id is not None:
            generate_payload["audio_asset_id"] = request.audio_asset_id
        
        logger.info(f"Calling HeyGen Avatar IV API with payload: {generate_payload}")
        
        generate_response = requests.post(
            generate_url,
            headers=generate_headers,
            json=generate_payload,
            timeout=120  # 2 minute timeout for API call
        )
        
        if generate_response.status_code != 200:
            logger.error(f"HeyGen Avatar IV generation failed: {generate_response.status_code} - {generate_response.text}")
            raise HTTPException(
                status_code=generate_response.status_code,
                detail=f"Failed to generate Avatar IV video: {generate_response.text}"
            )
        
        heygen_response = generate_response.json()
        logger.info(f"HeyGen Avatar IV video generation initiated successfully: {heygen_response}")
        
        # Return the HeyGen response directly
        return JSONResponse(status_code=200, content=heygen_response)
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error during HeyGen API call: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to HeyGen API: {str(e)}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during HeyGen Avatar IV generation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Avatar IV video: {str(e)}"
        )

@app.get("/telegram-file/bot{bot_token}/{file_path:path}")
async def telegram_file_to_s3(bot_token: str, file_path: str):
    """
    Download a Telegram file using bot token and file path, upload to S3 under audio/, and return the public URL.

    Telegram download URL format:
    https://api.telegram.org/file/bot<token>/<file_path>
    """
    logger.info(f"Telegram file request received - Path: {file_path}")

    try:
        # Validate required AWS environment variables
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500,
                detail="AWS credentials or bucket name not configured"
            )

        # Check if S3 client is available
        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )

        # Build Telegram file download URL
        telegram_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"
        logger.info(f"Downloading Telegram file from: {telegram_url}")

        try:
            download_response = requests.get(
                telegram_url,
                timeout=300,
                stream=True
            )
            download_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download Telegram file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download Telegram file: {str(e)}"
            )

        file_content = download_response.content
        if not file_content:
            logger.error("Downloaded Telegram file is empty")
            raise HTTPException(
                status_code=400,
                detail="Downloaded Telegram file is empty"
            )

        # Determine content type
        content_type = download_response.headers.get('Content-Type', 'application/octet-stream')
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()

        # Determine filename from file_path
        filename = os.path.basename(file_path)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"telegram_audio_{timestamp}_{unique_id}"

        # Ensure audio/ prefix
        s3_key = filename
        if not s3_key.startswith("audio/"):
            s3_key = f"audio/{s3_key}"

        logger.info(f"Uploading Telegram file to S3 - Bucket: {S3_BUCKET_NAME}, Key: {s3_key}")

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type,
            ACL='public-read'
        )

        logger.info(f"Telegram file uploaded to S3 successfully: {s3_key}")

        # Generate public URL
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Generated public URL: {public_url}")

        response_data = {
            "success": True,
            "message": "Telegram file downloaded and uploaded successfully",
            "public_url": public_url,
            "filename": filename,
            "s3_key": s3_key,
            "content_type": content_type,
            "uploaded_at": datetime.now().isoformat()
        }

        return JSONResponse(status_code=200, content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during Telegram file upload: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process Telegram file: {str(e)}"
        )

def extract_google_drive_file_id(file_id_or_url: str) -> str:
    """
    Extract Google Drive file ID from various URL formats or return the ID if already provided.
    
    Supported formats:
    - File ID only: "1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H"
    - Share link: "https://drive.google.com/file/d/1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H/view"
    - Download link: "https://drive.google.com/uc?id=1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H&export=download"
    - Open link: "https://drive.google.com/open?id=1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H"
    
    Args:
        file_id_or_url: Either a file ID or a full Google Drive URL
        
    Returns:
        The extracted file ID
        
    Raises:
        ValueError: If the file ID cannot be extracted
    """
    
    # Strip whitespace
    file_id_or_url = file_id_or_url.strip()
    
    # Check if it's a URL
    if file_id_or_url.startswith('http://') or file_id_or_url.startswith('https://'):
        parsed_url = urlparse(file_id_or_url)
        
        # Check if it's a Google Drive URL
        if 'drive.google.com' not in parsed_url.netloc and 'docs.google.com' not in parsed_url.netloc:
            raise ValueError(f"URL is not a Google Drive link: {file_id_or_url}")
        
        # Try to extract from query parameter 'id'
        # Format: https://drive.google.com/uc?id=FILE_ID&export=download
        # Format: https://drive.google.com/open?id=FILE_ID
        query_params = parse_qs(parsed_url.query)
        if 'id' in query_params:
            return query_params['id'][0]
        
        # Try to extract from path
        # Format: https://drive.google.com/file/d/FILE_ID/view
        # Format: https://drive.google.com/file/d/FILE_ID/edit
        path_match = re.search(r'/file/d/([a-zA-Z0-9_-]+)', parsed_url.path)
        if path_match:
            return path_match.group(1)
        
        # Try to extract from folders path
        # Format: https://drive.google.com/drive/folders/FILE_ID
        folder_match = re.search(r'/folders/([a-zA-Z0-9_-]+)', parsed_url.path)
        if folder_match:
            return folder_match.group(1)
        
        # Try to extract ID from any part of the URL using regex
        # Google Drive IDs are typically 28-33 characters with letters, numbers, underscores, and hyphens
        id_match = re.search(r'([a-zA-Z0-9_-]{25,})', file_id_or_url)
        if id_match:
            return id_match.group(1)
        
        raise ValueError(f"Could not extract file ID from URL: {file_id_or_url}")
    
    # Check if it looks like a valid Google Drive file ID
    # Google Drive IDs are typically alphanumeric with underscores and hyphens, 25+ characters
    if re.match(r'^[a-zA-Z0-9_-]{10,}$', file_id_or_url):
        return file_id_or_url
    
    raise ValueError(f"Invalid Google Drive file ID or URL: {file_id_or_url}")

@app.post("/google-drive-to-s3")
async def google_drive_to_s3(request: GoogleDriveToS3Request):
    """
    Download a publicly accessible file from Google Drive and upload it to AWS S3.
    
    This endpoint:
    1. Takes a Google Drive file ID or URL (various formats supported)
    2. Downloads the file with retry logic to handle large files and confirmation pages
    3. Uploads to S3 at the specified folder path
    4. Returns the public S3 URL
    
    Supported URL formats:
    - File ID only: "1Rs1rijuqphRVdxT6WDcPG86x9IMKG81H"
    - Share link: "https://drive.google.com/file/d/FILE_ID/view"
    - Download link: "https://drive.google.com/uc?id=FILE_ID&export=download"
    - Open link: "https://drive.google.com/open?id=FILE_ID"
    
    Note: The Google Drive file must be shared as "Anyone with the link can view"
    
    Args:
        request: GoogleDriveToS3Request containing:
            - file_id: Google Drive file ID or URL (ID will be extracted automatically)
            - folder_path: S3 folder path (e.g., "videos/", "images/")
            - filename: Optional filename (auto-detected if not provided)
    
    Returns:
        JSON response with the public S3 URL of the uploaded file
    """
    logger.info(f"Google Drive to S3 request received - Input: {request.file_id}, Folder: {request.folder_path}")
    
    try:
        # Extract file ID from URL or use as-is if already an ID
        try:
            file_id = extract_google_drive_file_id(request.file_id)
            logger.info(f"Extracted Google Drive file ID: {file_id}")
        except ValueError as e:
            logger.error(f"Failed to extract file ID: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
        
        # Validate required AWS environment variables
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
            logger.error("Missing required AWS environment variables")
            raise HTTPException(
                status_code=500,
                detail="AWS credentials or bucket name not configured"
            )
        
        # Check if S3 client is available
        if not s3_client:
            logger.error("S3 client not initialized")
            raise HTTPException(
                status_code=500,
                detail="S3 client not available - check AWS configuration"
            )
        
        # Normalize folder path (ensure it ends with / and doesn't start with /)
        folder_path = request.folder_path.strip('/')
        if folder_path:
            folder_path = folder_path + '/'
        
        # Google Drive download URL for publicly shared files
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Configure retry settings
        max_retries = 3
        retry_delay = 2  # seconds
        
        file_content = None
        filename = request.filename
        content_type = "application/octet-stream"
        
        # Create a session to handle cookies (needed for large file confirmation)
        session = requests.Session()
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Download attempt {attempt + 1}/{max_retries} for file ID: {file_id}")
                
                # Initial request
                response = session.get(
                    download_url,
                    stream=True,
                    timeout=30,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    }
                )
                response.raise_for_status()
                
                # Check if we got a confirmation page (for large files)
                # Google Drive returns HTML with a confirmation token for large files
                content_type_header = response.headers.get('Content-Type', '')
                
                if 'text/html' in content_type_header:
                    logger.info("Received HTML response, checking for download confirmation...")
                    
                    # Look for the confirmation token in the response
                    html_content = response.text
                    
                    # Look for confirm parameter in various formats
                    confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', html_content)
                    if confirm_match:
                        confirm_token = confirm_match.group(1)
                        logger.info(f"Found confirmation token: {confirm_token}")
                        
                        # Make the confirmed download request
                        confirmed_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                        response = session.get(
                            confirmed_url,
                            stream=True,
                            timeout=300,  # 5 minutes for large files
                            headers={
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                            }
                        )
                        response.raise_for_status()
                    else:
                        # Try alternative confirmation approach - look for uuid
                        uuid_match = re.search(r'uuid=([0-9A-Za-z_-]+)', html_content)
                        if uuid_match:
                            # Use the download_warning cookie approach
                            confirmed_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
                            response = session.get(
                                confirmed_url,
                                stream=True,
                                timeout=300,
                                headers={
                                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                                }
                            )
                            response.raise_for_status()
                        else:
                            logger.warning("Could not find confirmation token, file may not be downloadable")
                
                # Check content type again after potential redirect
                content_type_header = response.headers.get('Content-Type', 'application/octet-stream')
                if ';' in content_type_header:
                    content_type = content_type_header.split(';')[0].strip()
                else:
                    content_type = content_type_header
                
                # If we still have HTML, the file might not be publicly accessible
                if 'text/html' in content_type:
                    logger.error("File appears to not be publicly accessible or doesn't exist")
                    raise HTTPException(
                        status_code=400,
                        detail="Unable to download file. Please ensure the Google Drive file is shared as 'Anyone with the link can view'"
                    )
                
                # Try to get filename from Content-Disposition header if not provided
                if not filename:
                    content_disposition = response.headers.get('Content-Disposition', '')
                    if 'filename=' in content_disposition:
                        # Parse filename from header
                        filename_match = re.search(r'filename\*?=["\']?(?:UTF-8\'\')?([^"\';\s]+)["\']?', content_disposition)
                        if filename_match:
                            filename = unquote(filename_match.group(1))
                            logger.info(f"Extracted filename from header: {filename}")
                
                # Generate filename if still not available
                if not filename:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_id = str(uuid.uuid4())[:8]
                    
                    # Determine extension based on content type
                    extension_map = {
                        'image/jpeg': '.jpg',
                        'image/png': '.png',
                        'image/gif': '.gif',
                        'image/webp': '.webp',
                        'video/mp4': '.mp4',
                        'video/webm': '.webm',
                        'video/quicktime': '.mov',
                        'audio/mpeg': '.mp3',
                        'audio/wav': '.wav',
                        'audio/ogg': '.ogg',
                        'application/pdf': '.pdf',
                        'text/plain': '.txt',
                        'application/zip': '.zip',
                    }
                    extension = extension_map.get(content_type, '')
                    filename = f"gdrive_file_{timestamp}_{unique_id}{extension}"
                    logger.info(f"Generated filename: {filename}")
                
                # Download the file content
                logger.info("Downloading file content...")
                file_content = response.content
                content_length = len(file_content)
                
                if content_length == 0:
                    raise Exception("Downloaded file is empty")
                
                logger.info(f"File downloaded successfully. Size: {content_length} bytes ({content_length / (1024*1024):.2f} MB)")
                
                # Success - break out of retry loop
                break
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"All {max_retries} download attempts failed")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to download file from Google Drive after {max_retries} attempts: {str(e)}"
                    )
        
        if file_content is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to download file content"
            )
        
        # Build the full S3 key (folder path + filename)
        s3_key = f"{folder_path}{filename}"
        
        logger.info(f"Uploading to S3 - Bucket: {S3_BUCKET_NAME}, Key: {s3_key}")
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_content,
            ContentType=content_type,
            ACL='public-read'
        )
        
        logger.info(f"File uploaded to S3 successfully: {s3_key}")
        
        # Generate public URL
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Generated public URL: {public_url}")
        
        # Return success response
        response_data = {
            "success": True,
            "message": "File downloaded from Google Drive and uploaded to S3 successfully",
            "public_url": public_url,
            "filename": filename,
            "folder_path": folder_path,
            "s3_key": s3_key,
            "file_id": file_id,
            "original_input": request.file_id,
            "file_size_bytes": content_length,
            "content_type": content_type,
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info("Google Drive to S3 transfer completed successfully")
        return JSONResponse(status_code=200, content=response_data)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during Google Drive to S3 transfer: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transfer file from Google Drive to S3: {str(e)}"
        )

@app.post("/transcribe-audio")
async def transcribe_audio(
    request: TranscribeAudioRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Transcribe an audio file using OpenAI's transcription API.

    Requires:
    - Authorization: Bearer <OpenAI API key>
    - Request body with audio_url
    """
    logger.info(f"Transcription request received - Audio URL: {request.audio_url}")

    try:
        # Validate authorization header
        if not authorization or not authorization.lower().startswith("bearer "):
            logger.error("Missing or invalid authorization header")
            raise HTTPException(
                status_code=401,
                detail="Authorization header with Bearer token is required"
            )

        # Extract API key from authorization header
        api_key = authorization.split(" ", 1)[1].strip()

        # Download audio file
        logger.info(f"Downloading audio from URL: {request.audio_url}")
        try:
            audio_response = requests.get(
                request.audio_url,
                timeout=300,
                stream=True
            )
            audio_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download audio file: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to download audio file: {str(e)}"
            )

        audio_content = audio_response.content
        if not audio_content:
            logger.error("Downloaded audio file is empty")
            raise HTTPException(
                status_code=400,
                detail="Downloaded audio file is empty"
            )

        # Determine content type and filename
        content_type = audio_response.headers.get('Content-Type', 'application/octet-stream')
        if ';' in content_type:
            content_type = content_type.split(';')[0].strip()

        filename = None
        content_disposition = audio_response.headers.get('Content-Disposition', '')
        if 'filename=' in content_disposition:
            match = re.search(r'filename[*]?=["\']?([^"\';\s]+)["\']?', content_disposition)
            if match:
                filename = match.group(1)

        if not filename:
            parsed_url = urlparse(request.audio_url)
            path_filename = parsed_url.path.split('/')[-1]
            if path_filename:
                filename = unquote(path_filename)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = str(uuid.uuid4())[:8]
                filename = f"audio_{timestamp}_{unique_id}"

        # Call OpenAI transcription API
        openai_url = "https://api.openai.com/v1/audio/transcriptions"
        openai_headers = {
            "Authorization": f"Bearer {api_key}"
        }

        files = {
            "file": (filename, audio_content, content_type)
        }
        data = {
            "model": "whisper-1"
        }

        logger.info("Calling OpenAI transcription API")
        openai_response = requests.post(
            openai_url,
            headers=openai_headers,
            files=files,
            data=data,
            timeout=300
        )

        if openai_response.status_code != 200:
            logger.error(f"OpenAI API error: {openai_response.status_code} - {openai_response.text}")
            raise HTTPException(
                status_code=openai_response.status_code,
                detail=f"OpenAI API error: {openai_response.text}"
            )

        response_json = openai_response.json()
        transcript_text = response_json.get("text")

        if not transcript_text:
            logger.error(f"No transcript returned from OpenAI: {response_json}")
            raise HTTPException(
                status_code=500,
                detail="No transcript returned from OpenAI"
            )

        response_data = {
            "success": True,
            "message": "Transcription completed successfully",
            "transcript": transcript_text,
            "filename": filename,
            "content_type": content_type,
            "transcribed_at": datetime.now().isoformat()
        }

        return JSONResponse(status_code=200, content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to transcribe audio: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    logger.info(f"Environment check - AWS_REGION: {AWS_REGION}, S3_BUCKET: {S3_BUCKET_NAME}")
    logger.info(f"Initializing FastAPI app with AWS region: {AWS_REGION}")
    logger.info(f"S3 Bucket: {S3_BUCKET_NAME}")
    logger.info(f"AWS Access Key ID configured: {'Yes' if AWS_ACCESS_KEY_ID else 'No'}")
    logger.info(f"AWS Secret Access Key configured: {'Yes' if AWS_SECRET_ACCESS_KEY else 'No'}")
    logger.info(f"Assembly AI API Key configured: {'Yes' if ASSEMBLY_AI_API_KEY else 'No'}")
    logger.info(f"Environment variables loaded from .env file")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
