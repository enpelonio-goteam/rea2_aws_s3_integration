# REA Pro Utilities Provider

A FastAPI application that exposes a suite of HTTP utility endpoints used by REA Pro workflows: media downloads/uploads (S3, Google Drive, SharePoint, LinkedIn, Telegram), AI generation (OpenAI, Eleven Labs, HeyGen, AssemblyAI), and content conversion (HTML → image, Markdown → docx, image optimization). Deployable to Vercel.

## Features

- AWS S3 uploads with public URLs returned for downstream use
- AI generation: OpenAI image/transcription/vision, Eleven Labs TTS, HeyGen Avatar IV, AssemblyAI transcription
- Media pipelines: Loom video → S3 + transcript, LinkedIn UGC video posting, HeyGen avatar video
- Cross-storage transfers: URL → Google Drive, Google Drive → S3, URL → SharePoint, Telegram → S3
- Content conversion: HTML → PNG (Instagram dimensions), Markdown → docx, image compression to a max size
- SharePoint folder management (list subfolders, create folder) via Microsoft Graph app-only auth
- Vercel deployment ready

## Prerequisites

- Python 3.8+
- AWS S3 bucket with public read enabled (for endpoints that return public URLs)
- Provider credentials as required by the endpoint(s) you intend to call (see below)

## Environment Variables

```bash
# Required for any S3-backed endpoint
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name

# Required for /process-loom-video
ASSEMBLY_AI_API_KEY=your_assembly_ai_api_key

# Required for SharePoint endpoints (Azure AD app, application permission
# Sites.ReadWrite.All with admin consent)
SHAREPOINT_TENANT_ID=your_tenant_id
SHAREPOINT_CLIENT_ID=your_client_id
SHAREPOINT_CLIENT_SECRET=your_client_secret
```

Most third-party API keys (OpenAI, Eleven Labs, HeyGen, LinkedIn, Google Drive) are passed **per-request** via request headers rather than env vars — see the endpoint table below.

## Local Development

```bash
pip install -r requirements.txt
# Playwright is used by /html-to-image; install browsers if you call it locally:
# pip install playwright && playwright install chromium
python main.py
```

## API Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | API status, available endpoints, AWS configuration check |
| GET | `/health` | Health check |
| POST | `/upload-html` | Upload an HTML string to S3 under `html/`, return the public URL |
| POST | `/generate-image` | Generate an image with OpenAI (DALL-E / gpt-image-1) and store it in S3 |
| POST | `/process-loom-video` | Download a Loom video, push it to S3, and transcribe with AssemblyAI |
| POST | `/html-to-image` | Render HTML to a PNG (Instagram-friendly dimensions) via Playwright and upload to S3 |
| POST | `/optimize-image-to-s3` | Download an image URL and upload to S3, compressing to fit under `max_size_mb` if needed |
| POST | `/upload-linkedin-video` | Upload an MP4 from a URL to LinkedIn as a UGC post with caption and visibility |
| POST | `/url-to-google-drive` | Download a public URL and upload the file to a specific Google Drive folder |
| POST | `/eleven-labs-speech` | Generate speech via Eleven Labs TTS and upload the audio to S3 |
| POST | `/heygen-avatar-iv` | Upload an image to HeyGen and create an Avatar IV video (TTS or pre-recorded audio mode) |
| GET | `/telegram-file/bot{bot_token}/{file_path}` | Download a Telegram file via Bot API and upload to S3 under `audio/` |
| POST | `/google-drive-to-s3` | Download a publicly shared Google Drive file (accepts ID or URL) and upload to S3 |
| POST | `/transcribe-audio` | Transcribe an audio URL with OpenAI Whisper |
| POST | `/analyze-images` | Analyze multiple images (comma-separated URLs) with an OpenAI vision model |
| POST | `/markdown-to-docx` | Convert markdown to a `.docx` Word document and upload to S3 under `documents/` |
| POST | `/list-sharepoint-subfolders` | List immediate or recursive subfolders (and optionally files) of a SharePoint folder |
| POST | `/create-sharepoint-folder` | Create a subfolder inside a SharePoint folder via Microsoft Graph |
| POST | `/upload-to-sharepoint` | Download a public URL and upload the file to a SharePoint folder via Microsoft Graph |

### Authentication per endpoint

| Endpoint | Auth |
| --- | --- |
| `/generate-image`, `/transcribe-audio`, `/analyze-images`, `/url-to-google-drive` | `Authorization: Bearer <token>` (OpenAI key, or Google Drive OAuth access token) |
| `/upload-linkedin-video` | `Authorization: Bearer <linkedin_access_token>` (scopes: `r_liteprofile`, `w_member_social`) |
| `/eleven-labs-speech` | `xi-api-key: <eleven_labs_key>` header |
| `/heygen-avatar-iv` | `X-Api-Key: <heygen_key>` header |
| SharePoint endpoints (`/upload-to-sharepoint`, `/create-sharepoint-folder`, `/list-sharepoint-subfolders`) | App-only client-credentials via the `SHAREPOINT_*` env vars |
| `/process-loom-video` | AssemblyAI key from `ASSEMBLY_AI_API_KEY` env var |
| Others | No external auth required (only AWS env vars for S3-backed endpoints) |

## Usage Examples

### Upload HTML to S3

```bash
curl -X POST "$BASE_URL/upload-html" \
  -H "Content-Type: application/json" \
  -d '{"html_content": "<html><body><h1>Hello</h1></body></html>", "filename": "my-page.html"}'
```

### Process a Loom video (download → S3 → transcribe)

```bash
curl -X POST "$BASE_URL/process-loom-video" \
  -H "Content-Type: application/json" \
  -d '{"loom_url": "https://www.loom.com/share/<video_id>"}'
```

### Generate speech with Eleven Labs

```bash
curl -X POST "$BASE_URL/eleven-labs-speech" \
  -H "Content-Type: application/json" \
  -H "xi-api-key: $ELEVEN_LABS_KEY" \
  -d '{"voice_id": "21m00Tcm4TlvDq8ikWAM", "text": "Hello world"}'
```

### Convert markdown to docx

```bash
curl -X POST "$BASE_URL/markdown-to-docx" \
  -H "Content-Type: application/json" \
  -d '{"text": "# Title\n\nSome **bold** text.", "filename": "report"}'
```

### Upload a public file to SharePoint

```bash
curl -X POST "$BASE_URL/upload-to-sharepoint" \
  -H "Content-Type: application/json" \
  -d '{
    "file_url": "https://example.com/file.pdf",
    "sharepoint_url": "https://contoso.sharepoint.com/sites/Team/Shared%20Documents/Folder"
  }'
```

### Analyze multiple images with OpenAI vision

```bash
curl -X POST "$BASE_URL/analyze-images" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "Describe each image",
    "image_urls": "https://example.com/a.jpg,https://example.com/b.jpg",
    "model": "gpt-4o-mini"
  }'
```

Full request/response schemas live next to each handler in [main.py](main.py).

## AWS S3 Bucket Configuration

Endpoints that return public URLs upload objects with `ACL=public-read`. Your bucket must allow public-read for these objects.

1. Block Public Access: disabled (or scoped to allow object ACLs)
2. Bucket policy granting `s3:GetObject` to `*` for the bucket's objects:

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

S3 key prefixes used by the service: `html/`, `videos/`, `audio/`, `documents/`, `optimized-images/`, plus folders chosen per-request by `/google-drive-to-s3` and similar.

## Deployment to Vercel

```bash
npm i -g vercel
vercel
```

Set the environment variables above in the Vercel dashboard. See [DEPLOYMENT.md](DEPLOYMENT.md) for additional notes.

## Security Considerations

- Never commit AWS, SharePoint, or third-party API credentials
- Use IAM users / app registrations with the minimal permissions required
- Per-request bearer tokens (OpenAI, LinkedIn, Google Drive) are forwarded directly to the upstream provider and are not stored
- Monitor S3 access logs; consider lifecycle rules for the `videos/`, `audio/`, and `optimized-images/` prefixes

## License

MIT
