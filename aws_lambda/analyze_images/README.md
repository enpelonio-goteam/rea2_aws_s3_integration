# Analyze Images Lambda

This folder contains a standalone AWS Lambda function for image analysis with OpenAI.

## Runtime

- Python 3.11 recommended
- Lambda timeout: set to up to 900 seconds (15 minutes)
- Memory: start with 2048 MB

## Handler

- File: `lambda_function.py`
- Handler value: `lambda_function.lambda_handler`

## Request

Method: `POST`

Headers:
- `Authorization: Bearer <OPENAI_API_KEY>`
- `Content-Type: application/json`

Body:

```json
{
  "prompt": "Analyze these images",
  "image_urls": "https://example.com/a.jpg, https://example.com/b.jpg",
  "model": "gpt-4.1",
  "reasoning_effort": "medium"
}
```

## Response

Returns:
- `analysis`
- `raw_response`
- `payload_bytes`
- `timings_seconds`

## Deploy (zip)

From this folder (`aws_lambda/analyze_images`):

1. Install dependencies into a build dir:
   - `pip install -r requirements.txt -t package`
2. Copy handler:
   - `copy lambda_function.py package\\`
3. Zip contents of `package` directory and upload to Lambda.

## Notes

- The function downloads images in parallel, optimizes them in parallel, and submits one OpenAI request.
- OpenAI timeout is controlled by environment variable `OPENAI_TIMEOUT_SECONDS` (default `600`).
