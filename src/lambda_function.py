import json
import boto3
import os
import logging
from datetime import datetime
import uuid
from typing import Dict, Any
from urllib.parse import unquote_plus

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Initialize AWS clients for reuse
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')
comprehend_client = boto3.client('comprehend')

# Get environment variables
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
SNS_TOPIC = os.environ['SNS_TOPIC']
# The KMS_KEY_ID is used via the Lambda's IAM role permissions
KMS_KEY_ID = os.environ['KMS_KEY_ID'] 

# DynamoDB table reference
table = dynamodb.Table(DYNAMODB_TABLE)

# --- Custom Exceptions ---
class FileValidationError(Exception):
    """Custom exception for file validation errors."""
    pass

class ProcessingError(Exception):
    """Custom exception for general processing errors."""
    pass

# --- Main Handler ---
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing text files uploaded to S3.
    """
    logger.info(f"Processing event: {json.dumps(event, default=str)}")
    
    results = []
    for record in event['Records']:
        try:
            result = process_s3_record(record)
            results.append(result)
            logger.info(f"Successfully processed record: {result}")
        except (FileValidationError, ProcessingError) as e:
            logger.warning(f"Skipping record due to validation/processing error: {str(e)}")
            results.append({'status': 'skipped', 'reason': str(e)})
        except Exception as e:
            logger.error(f"Critical error processing record: {str(e)}")
            results.append({'status': 'error', 'error': str(e)})
            # In a real-world scenario, you might want to send this to a Dead Letter Queue (DLQ)
            
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Processing completed',
            'results': results
        })
    }

# --- Core Logic ---
def process_s3_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single S3 record from the event.
    """
    bucket = record['s3']['bucket']['name']
    key = unquote_plus(record['s3']['object']['key'])
    size = record['s3']['object']['size']
    
    logger.info(f"Processing file: {key} from bucket: {bucket}")

    if not is_valid_file_type(key):
        raise FileValidationError(f"Unsupported file type for '{key}'. Supported types are .txt, .json, .md.")

    file_content = read_s3_file(bucket, key)
    if not file_content or len(file_content.strip()) < 10: # Require at least 10 characters
        raise FileValidationError(f"File '{key}' is empty or content is too short for analysis.")
    
    # Truncate content if too long for Comprehend's 5000 byte limit
    if len(file_content.encode('utf-8')) > 4900:
        file_content = file_content[:4500]
        logger.warning(f"File content truncated for analysis: {key}")

    language_code = get_dominant_language(file_content)
    sentiment_result = analyze_sentiment(file_content, language_code)
    
    metadata = {
        'fileId': str(uuid.uuid4()),
        'fileName': key,
        'fileSize': size,
        'fileType': get_file_extension(key),
        'uploadTime': datetime.utcnow().isoformat(),
        'sentiment': sentiment_result['Sentiment'],
        'sentimentScore': sentiment_result['SentimentScore'],
        'detectedLanguage': language_code
    }
    
    store_metadata(metadata)
    send_notification(metadata)
    
    return {
        'status': 'success',
        'fileId': metadata['fileId'],
        'fileName': key,
        'sentiment': metadata['sentiment']
    }

# --- Helper Functions ---
def is_valid_file_type(key: str) -> bool:
    """Check if the file type is supported."""
    return any(key.lower().endswith(ext) for ext in ['.txt', '.json', '.md'])

def get_file_extension(key: str) -> str:
    """Get file extension from S3 key."""
    return os.path.splitext(key)[1].lower()

def read_s3_file(bucket: str, key: str) -> str:
    """Read file content from S3."""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        logger.error(f"Error reading file {key} from bucket {bucket}: {str(e)}")
        raise ProcessingError(f"Could not read file from S3: {key}") from e

def get_dominant_language(text: str) -> str:
    """Detect the dominant language using Amazon Comprehend."""
    try:
        response = comprehend_client.detect_dominant_language(Text=text)
        language = max(response['Languages'], key=lambda x: x['Score'])
        return language['LanguageCode']
    except Exception as e:
        logger.warning(f"Could not detect language: {e}. Defaulting to 'en'.")
        return 'en'

def analyze_sentiment(text: str, language_code: str) -> Dict[str, Any]:
    """Analyze sentiment using Amazon Comprehend."""
    try:
        response = comprehend_client.detect_sentiment(Text=text, LanguageCode=language_code)
        logger.info(f"Sentiment analysis completed: {response['Sentiment']}")
        return response
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise ProcessingError("Failed to analyze sentiment.") from e

def store_metadata(metadata: Dict[str, Any]) -> None:
    """Store file metadata in DynamoDB."""
    try:
        # Boto3 DynamoDB Resource automatically handles Python floats to DynamoDB Decimals
        table.put_item(Item=metadata)
        logger.info(f"Metadata stored successfully for file: {metadata['fileName']}")
    except Exception as e:
        logger.error(f"Error storing metadata: {str(e)}")
        raise ProcessingError("Failed to store metadata in DynamoDB.") from e

def send_notification(metadata: Dict[str, Any]) -> None:
    """Send processing notification via SNS."""
    try:
        subject = f"File Processing Complete: {metadata['fileName']}"
        sentiment = metadata['sentiment']
        confidence = max(metadata['sentimentScore'].values())
        
        message = (
            f"File Processing Summary:\n"
            f"========================\n\n"
            f"File: {metadata['fileName']}\n"
            f"File Size: {metadata['fileSize']} bytes\n"
            f"Detected Language: {metadata['detectedLanguage'].upper()}\n\n"
            f"Sentiment Analysis:\n"
            f"- Result: {sentiment}\n"
            f"- Confidence: {confidence:.2%}\n\n"
            f"File ID: {metadata['fileId']}\n"
        )
        
        sns_client.publish(
            TopicArn=SNS_TOPIC,
            Message=message,
            Subject=subject
        )
        logger.info(f"Notification sent for file: {metadata['fileName']}")
    except Exception as e:
        # Log error but don't fail the entire process if notification fails
        logger.error(f"Error sending SNS notification: {str(e)}")