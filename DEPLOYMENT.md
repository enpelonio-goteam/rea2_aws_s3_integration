# Deployment Guide for Vercel

This guide will walk you through deploying your FastAPI application to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally with `npm i -g vercel`
3. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)

## Step 1: Prepare Your AWS S3 Bucket

### 1.1 Create S3 Bucket
- Go to AWS S3 Console
- Create a new bucket with a unique name
- Choose your preferred region
- **Important**: Uncheck "Block all public access" during creation

### 1.2 Configure Bucket Policy
Add this bucket policy to allow public read access:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::YOUR_BUCKET_NAME/*"
    }
  ]
}
```

### 1.3 Create IAM User
1. Go to AWS IAM Console
2. Create a new user with programmatic access
3. Attach the `AmazonS3FullAccess` policy (or create a custom policy with minimal permissions)
4. Save the Access Key ID and Secret Access Key

## Step 2: Deploy to Vercel

### 2.1 Login to Vercel
```bash
vercel login
```

### 2.2 Deploy Your Project
```bash
vercel
```

Follow the prompts:
- Set up and deploy? → `Y`
- Which scope? → Select your account
- Link to existing project? → `N`
- Project name? → Enter a name (e.g., `html-s3-uploader`)
- Directory? → `.` (current directory)
- Override settings? → `N`

### 2.3 Set Environment Variables
After deployment, set your environment variables:

```bash
vercel env add AWS_ACCESS_KEY_ID
vercel env add AWS_SECRET_ACCESS_KEY
vercel env add AWS_REGION
vercel env add S3_BUCKET_NAME
```

Or use the Vercel dashboard:
1. Go to your project in Vercel dashboard
2. Click on "Settings" → "Environment Variables"
3. Add each variable:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
   - `AWS_REGION`: Your S3 bucket region (e.g., `us-east-1`)
   - `S3_BUCKET_NAME`: Your S3 bucket name

### 2.4 Redeploy with Environment Variables
```bash
vercel --prod
```

## Step 3: Test Your Deployment

### 3.1 Test the Health Endpoint
```bash
curl https://your-app.vercel.app/health
```

### 3.2 Test HTML Upload
```bash
curl -X POST "https://your-app.vercel.app/upload-html" \
  -H "Content-Type: application/json" \
  -d '{
    "html_content": "<html><body><h1>Test</h1></body></html>",
    "filename": "test.html"
  }'
```

## Step 4: Custom Domain (Optional)

1. In Vercel dashboard, go to "Settings" → "Domains"
2. Add your custom domain
3. Follow DNS configuration instructions

## Troubleshooting

### Common Issues

1. **Environment Variables Not Set**
   - Ensure all environment variables are set in Vercel
   - Redeploy after setting variables

2. **AWS Permissions Error**
   - Check IAM user has proper S3 permissions
   - Verify bucket name and region are correct

3. **CORS Issues**
   - Add CORS configuration to your S3 bucket if needed

4. **Function Timeout**
   - The `vercel.json` sets max duration to 30 seconds
   - Increase if needed for large uploads

### Debugging

1. **Check Vercel Logs**
   ```bash
   vercel logs
   ```

2. **Local Testing**
   - Test locally first with `python main.py`
   - Use the `test_api.py` script

3. **AWS CloudTrail**
   - Enable CloudTrail to monitor S3 API calls
   - Check for permission denied errors

## Security Best Practices

1. **IAM Permissions**: Use least privilege principle
2. **Environment Variables**: Never commit secrets to Git
3. **Bucket Policies**: Regularly review and audit
4. **Monitoring**: Set up CloudWatch alerts for unusual activity

## Cost Optimization

1. **S3 Storage**: Monitor storage usage
2. **Data Transfer**: Consider CloudFront for high-traffic scenarios
3. **Vercel Functions**: Monitor function execution time and memory usage

## Support

- **Vercel Documentation**: [vercel.com/docs](https://vercel.com/docs)
- **AWS S3 Documentation**: [docs.aws.amazon.com/s3](https://docs.aws.amazon.com/s3)
- **FastAPI Documentation**: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
