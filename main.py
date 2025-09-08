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
        "endpoints": ["/upload-html", "/generate-image", "/process-loom-video", "/health"],
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
