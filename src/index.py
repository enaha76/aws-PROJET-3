import json
import boto3
import os
from datetime import datetime
import urllib.parse
import logging
from botocore.exceptions import ClientError

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')
bedrock_runtime = boto3.client('bedrock-runtime')

# Environment variables
TABLE_NAME = os.environ['TABLE_NAME']
TOPIC_ARN = os.environ['TOPIC_ARN']
BEDROCK_MODEL_ID = os.environ.get('BEDROCK_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0')

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Process uploaded files for sentiment analysis
    """
    try:
        # Extract S3 event details
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
        
        logger.info(f"Processing file: {key} from bucket: {bucket}")
        
        # Download and read file
        file_content = download_file(bucket, key)
        
        # Extract text based on file type
        if key.lower().endswith('.pdf'):
            text_content = extract_text_from_pdf(file_content)
        else:
            text_content = file_content.decode('utf-8')
        
        # Perform sentiment analysis with Bedrock
        analysis_result = analyze_sentiment_with_bedrock(text_content)
        
        # Store metadata in DynamoDB
        save_to_dynamodb(key, analysis_result)
        
        # Send notification
        send_notification(key, analysis_result)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'File processed successfully',
                'file': key,
                'sentiment': analysis_result['sentiment']
            })
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        
        # Send error notification
        send_error_notification(key if 'key' in locals() else 'Unknown', str(e))
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def download_file(bucket, key):
    """Download file from S3"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except ClientError as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF content"""
    try:
        import PyPDF2
        from io import BytesIO
        
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        # Fallback to basic extraction
        return pdf_content.decode('utf-8', errors='ignore')

def analyze_sentiment_with_bedrock(text):
    """Analyze sentiment using Bedrock Claude model"""
    try:
        # Prepare the prompt
        prompt = f"""
        Please analyze the sentiment of the following text and provide:
        1. Overall sentiment (positive, negative, or neutral)
        2. Confidence score (0-100%)
        3. Key themes identified
        4. A brief summary (2-3 sentences)
        5. Main emotions detected
        
        Format the response as JSON with the following structure:
        {{
            "sentiment": "positive/negative/neutral",
            "confidence": 85,
            "themes": ["theme1", "theme2"],
            "summary": "Brief summary of the content",
            "emotions": ["emotion1", "emotion2"],
            "key_points": ["point1", "point2"]
        }}
        
        Text to analyze:
        {text[:3000]}  # Limit text to avoid token limits
        """
        
        # Call Bedrock
        response = bedrock_runtime.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType='application/json',
            accept='application/json',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.9
            })
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        
        # Extract the JSON from Claude's response
        content = response_body.get('content', [{}])[0].get('text', '{}')
        
        # Find JSON in the response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
        else:
            # Fallback if JSON extraction fails
            analysis = {
                "sentiment": "neutral",
                "confidence": 50,
                "themes": ["Unable to extract themes"],
                "summary": "Analysis completed but format parsing failed",
                "emotions": ["unknown"],
                "key_points": ["Review required"]
            }
        
        # Add metadata
        analysis['processed_at'] = datetime.utcnow().isoformat()
        analysis['model_used'] = BEDROCK_MODEL_ID
        analysis['text_length'] = len(text)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error calling Bedrock: {e}")
        return {
            "sentiment": "error",
            "confidence": 0,
            "themes": [],
            "summary": f"Error during analysis: {str(e)}",
            "emotions": [],
            "key_points": [],
            "error": str(e)
        }

def save_to_dynamodb(filename, analysis_result):
    """Save analysis results to DynamoDB"""
    try:
        table = dynamodb.Table(TABLE_NAME)
        
        item = {
            'FileName': filename,
            'ProcessedAt': datetime.utcnow().isoformat(),
            'Sentiment': analysis_result['sentiment'],
            'Confidence': analysis_result['confidence'],
            'Themes': analysis_result.get('themes', []),
            'Summary': analysis_result.get('summary', ''),
            'Emotions': analysis_result.get('emotions', []),
            'KeyPoints': analysis_result.get('key_points', []),
            'TextLength': analysis_result.get('text_length', 0),
            'ModelUsed': analysis_result.get('model_used', BEDROCK_MODEL_ID)
        }
        
        table.put_item(Item=item)
        logger.info(f"Saved analysis results to DynamoDB for file: {filename}")
        
    except Exception as e:
        logger.error(f"Error saving to DynamoDB: {e}")
        raise

def send_notification(filename, analysis_result):
    """Send email notification with analysis results"""
    try:
        subject = f"Sentiment Analysis Complete: {filename}"
        
        message = f"""
        File Analysis Complete!
        
        File: {filename}
        Processed: {analysis_result.get('processed_at', 'N/A')}
        
        === SENTIMENT ANALYSIS RESULTS ===
        
        Overall Sentiment: {analysis_result['sentiment'].upper()}
        Confidence: {analysis_result['confidence']}%
        
        === SUMMARY ===
        {analysis_result.get('summary', 'No summary available')}
        
        === KEY THEMES ===
        {', '.join(analysis_result.get('themes', ['No themes identified']))}
        
        === EMOTIONS DETECTED ===
        {', '.join(analysis_result.get('emotions', ['No emotions detected']))}
        
        === KEY POINTS ===
        {chr(10).join(['• ' + point for point in analysis_result.get('key_points', ['No key points identified'])])}
        
        === METADATA ===
        Text Length: {analysis_result.get('text_length', 0)} characters
        Model Used: {analysis_result.get('model_used', 'Unknown')}
        
        This analysis was performed using Amazon Bedrock with Claude AI.
        """
        
        sns_client.publish(
            TopicArn=TOPIC_ARN,
            Subject=subject,
            Message=message
        )
        
        logger.info(f"Notification sent for file: {filename}")
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise

def send_error_notification(filename, error_message):
    """Send error notification"""
    try:
        subject = f"Error Processing File: {filename}"
        message = f"""
        An error occurred while processing the file.
        
        File: {filename}
        Error: {error_message}
        Time: {datetime.utcnow().isoformat()}
        
        Please check CloudWatch logs for more details.
        """
        
        sns_client.publish(
            TopicArn=TOPIC_ARN,
            Subject=subject,
            Message=message
        )
    except Exception as e:
        logger.error(f"Error sending error notification: {e}")