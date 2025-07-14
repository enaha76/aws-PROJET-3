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

# Initialize AWS clients
s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
sns_client = boto3.client('sns')
comprehend_client = boto3.client('comprehend')

# Get environment variables
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE']
SNS_TOPIC = os.environ['SNS_TOPIC']

# DynamoDB table reference
table = dynamodb.Table(DYNAMODB_TABLE)

def lambda_handler(event: Dict[str, Any], context: Any) -> None:
    """
    Main Lambda handler for processing text files uploaded to S3.
    """
    for record in event['Records']:
        try:
            process_s3_record(record)
        except Exception as e:
            logger.error(f"Error processing record: {str(e)}", exc_info=True)
            # Future enhancement: Send failed record to the Dead Letter Queue

def process_s3_record(record: Dict[str, Any]) -> None:
    """
    Process a single S3 record from the event.
    """
    bucket = record['s3']['bucket']['name']
    key = unquote_plus(record['s3']['object']['key'])
    
    logger.info(f"Processing file: {key} from bucket: {bucket}")

    # Read file content from S3
    file_content = read_s3_file(bucket, key)
    
    # Analyze sentiment
    sentiment_result = analyze_sentiment(file_content)
    
    # Prepare metadata
    metadata = {
        'fileId': str(uuid.uuid4()),
        'fileName': key,
        'fileSize': record['s3']['object']['size'],
        'uploadTime': datetime.utcnow().isoformat(),
        'sentiment': sentiment_result['Sentiment'],
        'sentimentScore': sentiment_result['SentimentScore']
    }
    
    # Store metadata and send notification
    store_metadata(metadata)
    send_notification(metadata)

def read_s3_file(bucket: str, key: str) -> str:
    """Read file content from S3."""
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment using Amazon Comprehend."""
    # Truncate text if too long for Comprehend's 5000 byte limit
    if len(text.encode('utf-8')) > 4900:
        text = text[:4500]
    
    response = comprehend_client.detect_sentiment(Text=text, LanguageCode='en')
    return response

def store_metadata(metadata: Dict[str, Any]) -> None:
    """Store file metadata in DynamoDB."""
    # Boto3 DynamoDB Resource automatically handles Python floats to DynamoDB Decimals
    table.put_item(Item=metadata)
    logger.info(f"Metadata stored for file: {metadata['fileName']}")

def send_notification(metadata: Dict[str, Any]) -> None:
    """Send processing notification via SNS."""
    subject = f"File Processing Complete: {metadata['fileName']}"
    sentiment = metadata['sentiment']
    confidence = max(metadata['sentimentScore'].values())
    
    message = (
        f"File Processing Summary:\n\n"
        f"File: {metadata['fileName']}\n"
        f"Sentiment: {sentiment}\n"
        f"Confidence: {confidence:.2%}\n"
        f"File ID: {metadata['fileId']}"
    )
    
    sns_client.publish(
        TopicArn=SNS_TOPIC,
        Message=message,
        Subject=subject
    )
    logger.info(f"Notification sent for file: {metadata['fileName']}")