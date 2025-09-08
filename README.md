# Logic Provider Functions - FastAPI Project

A FastAPI application that provides various utility functions including HTML uploads, AI image generation, and Loom video processing with transcription. This project is deployable to Vercel.

## Features

- Upload HTML content to AWS S3 bucket
- Generate AI images using OpenAI DALL-E
- Process Loom videos: download, upload to S3, and transcribe with Assembly AI
- Automatic filename generation with timestamps
- Returns public URLs for uploaded content
- Health check endpoint
- Deployable to Vercel

## Prerequisites

- AWS S3 bucket configured with public access
- AWS credentials (Access Key ID and Secret Access Key)
- Assembly AI API key for transcription services
- OpenAI API key for image generation (optional)
- Python 3.8+

## Environment Variables

Set the following environment variables in your Vercel deployment:

```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=your_bucket_region (default: us-east-1)
S3_BUCKET_NAME=your_bucket_name
ASSEMBLY_AI_API_KEY=your_assembly_ai_api_key
OPENAI_API_KEY=your_openai_api_key (optional, for image generation)
```

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set environment variables
4. Run the application:
   ```bash
   python main.py
   ```

## API Endpoints

### Root Endpoint
- **GET** `/` - Returns API status

### Upload HTML
- **POST** `/upload-html`
  - **Body**: JSON with `html_content` and optional `filename`
  - **Returns**: Upload status and public URL

### Generate Image
- **POST** `/generate-image`
  - **Headers**: `Authorization: Bearer <openai_api_key>`
  - **Body**: JSON with `prompt` and optional `size`, `quality`
  - **Returns**: Generated image URL and metadata

### Process Loom Video
- **POST** `/process-loom-video`
  - **Body**: JSON with `loom_url` and optional `filename`
  - **Returns**: Video URL, transcription text, and detailed metadata

### Health Check
- **GET** `/health` - Returns health status

## Usage Examples

### Upload HTML Content

```bash
curl -X POST "https://your-vercel-app.vercel.app/upload-html" \
  -H "Content-Type: application/json" \
  -d '{
    "html_content": "<html><body><h1>Hello World</h1></body></html>",
    "filename": "my-page.html"
  }'
```

### Response Example

```json
{
  "success": true,
  "message": "HTML content uploaded successfully",
  "filename": "html/my-page.html",
  "public_url": "https://your-bucket.s3.us-east-1.amazonaws.com/html/my-page.html",
  "bucket": "your-bucket",
  "region": "us-east-1",
  "key": "html/my-page.html",
  "uploaded_at": "2024-01-15T10:30:00.000000"
}
```

### Process Loom Video

```bash
curl -X POST "https://your-vercel-app.vercel.app/process-loom-video" \
  -H "Content-Type: application/json" \
  -d '{
    "loom_url": "https://www.loom.com/share/your-video-id",
    "filename": "my-loom-video.mp4"
  }'
```

### Response Example

```json
{
  "success": true,
  "message": "Video processed and transcribed successfully",
  "video_url": "https://your-bucket.s3.us-east-1.amazonaws.com/videos/my-loom-video.mp4",
  "transcript": {
    "text": "This is the full transcription text...",
    "confidence": 0.95,
    "duration": 120.5,
    "language": "en",
    "words": [
      {
        "text": "This",
        "start": 0.0,
        "end": 0.5,
        "confidence": 0.99
      }
    ]
  },
  "filename": "videos/my-loom-video.mp4",
  "processed_at": "2024-01-15T10:30:00.000000"
}
```

### Generate Image

```bash
curl -X POST "https://your-vercel-app.vercel.app/generate-image" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-openai-api-key" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "size": "1024x1024",
    "quality": "standard"
  }'
```

## AWS S3 Bucket Configuration

Ensure your S3 bucket has the following settings:

1. **Block Public Access**: Disabled (to allow public read access)
2. **Bucket Policy**: Configured to allow public read access
3. **CORS Configuration**: If needed for web applications

### Example Bucket Policy

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::your-bucket-name/*"
    }
  ]
}
```

## Deployment to Vercel

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Deploy:
   ```bash
   vercel
   ```

3. Set environment variables in Vercel dashboard

## Security Considerations

- Never commit AWS credentials to version control
- Use IAM roles with minimal required permissions
- Consider using AWS STS for temporary credentials
- Monitor S3 bucket access logs

## License

MIT
