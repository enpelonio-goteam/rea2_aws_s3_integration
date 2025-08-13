# HTML to S3 Uploader - FastAPI Project

A FastAPI application that allows you to upload HTML content to an AWS S3 bucket and returns a public URL. This project is deployable to Vercel.

## Features

- Upload HTML content to AWS S3 bucket
- Automatic filename generation with timestamps
- Returns public URL for uploaded content
- Health check endpoint
- Deployable to Vercel

## Prerequisites

- AWS S3 bucket configured with public access
- AWS credentials (Access Key ID and Secret Access Key)
- Python 3.8+

## Environment Variables

Set the following environment variables in your Vercel deployment:

```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_REGION=your_bucket_region (default: us-east-1)
S3_BUCKET_NAME=your_bucket_name
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
  "filename": "my-page.html",
  "public_url": "https://your-bucket.s3.us-east-1.amazonaws.com/my-page.html",
  "bucket": "your-bucket",
  "region": "us-east-1",
  "key": "my-page.html",
  "uploaded_at": "2024-01-15T10:30:00.000000"
}
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
