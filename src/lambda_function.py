#!/usr/bin/env python3
"""
AWS Lambda Function for Text Processing Pipeline
Project: PROJET-3-GROUP-21029-21076-21047-24265

This function processes text files uploaded to S3, performs sentiment analysis
using Amazon Comprehend, stores metadata in DynamoDB, and sends notifications via SNS.

Author: AWS Architecture Team
Version: 1.0.0
"""

import json
import boto3
import os
import logging
import traceback
from datetime import datetime, timezone
import uuid
from typing import Dict, Any, List, Optional
from urllib.parse import unquote_plus
import re
from decimal import Decimal
from botocore.exceptions import ClientError, BotoCoreError

# Configure logging
logger = logging.getLogger()
logger.setLevel(os.environ.get('LOG_LEVEL', 'INFO'))

# Disable boto3 debug logging
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)

# Initialize AWS clients with error handling
try:
    s3_client = boto3.client('s3')
    dynamodb = boto3.resource('dynamodb')
    sns_client = boto3.client('sns')
    comprehend_client = boto3.client('comprehend')
except Exception as e:
    logger.error(f"Failed to initialize AWS clients: {str(e)}")
    raise

# Environment variables with defaults
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE')
SNS_TOPIC = os.environ.get('SNS_TOPIC')
KMS_KEY_ID = os.environ.get('KMS_KEY_ID')
PROJECT_NAME = os.environ.get('PROJECT_NAME', 'PROJET-3-GROUP-21029-21076-21047-24265')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')

# Validate required environment variables
required_env_vars = ['DYNAMODB_TABLE', 'SNS_TOPIC']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {missing_vars}")

# DynamoDB table reference
try:
    table = dynamodb.Table(DYNAMODB_TABLE)
except Exception as e:
    logger.error(f"Failed to initialize DynamoDB table: {str(e)}")
    raise

# Constants
SUPPORTED_FILE_EXTENSIONS = ['.txt', '.json', '.md']
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_COMPREHEND_TEXT_LENGTH = 5000  # Amazon Comprehend limit
TRUNCATE_LENGTH = 4800  # Leave buffer for safety

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for processing text files uploaded to S3.
    
    Args:
        event: S3 event trigger containing file upload information
        context: Lambda runtime context
        
    Returns:
        Dict containing processing results and status
    """
    # Log function start
    request_id = context.aws_request_id if context else str(uuid.uuid4())
    logger.info(f"[{request_id}] Lambda function started")
    logger.info(f"[{request_id}] Processing event with {len(event.get('Records', []))} records")
    
    start_time = datetime.now(timezone.utc)
    results = []
    errors = []
    
    try:
        # Validate event structure
        if 'Records' not in event:
            raise ProcessingError("Invalid event structure: missing 'Records' field")
        
        if not event['Records']:
            logger.warning(f"[{request_id}] No records to process")
            return create_response(200, "No records to process", [], [])
        
        # Process each record in the event
        for i, record in enumerate(event['Records']):
            record_id = f"{request_id}-{i}"
            logger.info(f"[{record_id}] Processing record {i+1}/{len(event['Records'])}")
            
            try:
                result = process_s3_record(record, record_id)
                results.append(result)
                logger.info(f"[{record_id}] Successfully processed: {result['fileName']}")
                
            except Exception as e:
                error_msg = str(e)
                error_details = {
                    'record_index': i,
                    'record_id': record_id,
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'file_key': get_safe_file_key(record),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                errors.append(error_details)
                logger.error(f"[{record_id}] Error processing record: {error_msg}")
                logger.error(f"[{record_id}] Error traceback: {traceback.format_exc()}")
                
                # Continue processing other records
                continue
        
        # Calculate processing summary
        end_time = datetime.now(timezone.utc)
        processing_duration = (end_time - start_time).total_seconds()
        
        logger.info(f"[{request_id}] Processing completed in {processing_duration:.2f}s")
        logger.info(f"[{request_id}] Results: {len(results)} successful, {len(errors)} errors")
        
        # Determine response status
        if errors and not results:
            status_code = 500  # All failed
        elif errors:
            status_code = 207  # Partial success
        else:
            status_code = 200  # All successful
        
        return create_response(
            status_code,
            f"Processed {len(results)} files successfully, {len(errors)} errors",
            results,
            errors,
            processing_duration
        )
        
    except Exception as e:
        end_time = datetime.now(timezone.utc)
        processing_duration = (end_time - start_time).total_seconds()
        
        logger.error(f"[{request_id}] Critical error in lambda_handler: {str(e)}")
        logger.error(f"[{request_id}] Critical error traceback: {traceback.format_exc()}")
        
        return create_response(
            500,
            f"Critical processing error: {str(e)}",
            results,
            errors + [{'error': str(e), 'error_type': 'CriticalError'}],
            processing_duration
        )

def create_response(status_code: int, message: str, results: List[Dict], errors: List[Dict], duration: float = 0) -> Dict[str, Any]:
    """Create standardized Lambda response"""
    return {
        'statusCode': status_code,
        'body': json.dumps({
            'message': message,
            'summary': {
                'successful_files': len(results),
                'failed_files': len(errors),
                'processing_duration_seconds': round(duration, 2),
                'timestamp': datetime.now(timezone.utc).isoformat()
            },
            'results': results,
            'errors': errors,
            'project': PROJECT_NAME,
            'environment': ENVIRONMENT
        }, default=str)
    }

def get_safe_file_key(record: Dict[str, Any]) -> str:
    """Safely extract file key from S3 record"""
    try:
        return record.get('s3', {}).get('object', {}).get('key', 'unknown')
    except:
        return 'unknown'

def process_s3_record(record: Dict[str, Any], record_id: str) -> Dict[str, Any]:
    """
    Process a single S3 record from the event.
    
    Args:
        record: S3 event record
        record_id: Unique identifier for this record
        
    Returns:
        Dict with processing result
    """
    logger.info(f"[{record_id}] Starting S3 record processing")
    
    # Extract and validate S3 information
    s3_info = extract_s3_info(record)
    bucket = s3_info['bucket']
    key = s3_info['key']
    size = s3_info['size']
    
    logger.info(f"[{record_id}] Processing file: {key} ({size} bytes) from bucket: {bucket}")
    
    # Validate file
    validate_file(key, size, record_id)
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    logger.info(f"[{record_id}] Generated file ID: {file_id}")
    
    # Read file content from S3
    file_content = read_s3_file(bucket, key, record_id)
    
    # Validate and prepare content for analysis
    processed_content = prepare_content_for_analysis(file_content, key, record_id)
    
    # Perform AI analysis
    sentiment_result = analyze_sentiment(processed_content, record_id)
    entities_result = extract_entities(processed_content, record_id)
    language_result = detect_language(processed_content, record_id)
    
    # Prepare comprehensive metadata
    metadata = create_metadata(
        file_id, key, size, bucket, file_content, 
        sentiment_result, entities_result, language_result, record_id
    )
    
    # Store metadata in DynamoDB
    store_metadata(metadata, record_id)
    
    # Send notification
    send_notification(metadata, record_id)
    
    logger.info(f"[{record_id}] Successfully completed processing")
    
    return {
        'status': 'success',
        'fileId': file_id,
        'fileName': key,
        'fileSize': size,
        'sentiment': sentiment_result['Sentiment'],
        'confidence': round(max(sentiment_result['SentimentScore'].values()), 3),
        'language': language_result.get('LanguageCode', 'unknown'),
        'entities_count': len(entities_result),
        'processing_timestamp': datetime.now(timezone.utc).isoformat()
    }

def extract_s3_info(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract S3 information from event record"""
    try:
        s3_record = record['s3']
        return {
            'bucket': s3_record['bucket']['name'],
            'key': unquote_plus(s3_record['object']['key']),
            'size': s3_record['object']['size'],
            'etag': s3_record['object'].get('eTag', ''),
            'version_id': s3_record['object'].get('versionId', '')
        }
    except KeyError as e:
        raise ProcessingError(f"Invalid S3 record structure: missing {str(e)}")

def validate_file(key: str, size: int, record_id: str) -> None:
    """
    Validate file type and size.
    
    Args:
        key: S3 object key
        size: File size in bytes
        record_id: Record identifier for logging
    """
    logger.info(f"[{record_id}] Validating file: {key}")
    
    # Check file extension
    if not is_valid_file_type(key):
        supported = ', '.join(SUPPORTED_FILE_EXTENSIONS)
        raise FileValidationError(f"Unsupported file type: {key}. Supported types: {supported}")
    
    # Check file size
    if size > MAX_FILE_SIZE:
        raise FileValidationError(f"File too large: {size} bytes. Maximum: {MAX_FILE_SIZE} bytes")
    
    if size == 0:
        raise FileValidationError("File is empty")
    
    logger.info(f"[{record_id}] File validation successful")

def is_valid_file_type(key: str) -> bool:
    """Check if the file type is supported"""
    return any(key.lower().endswith(ext) for ext in SUPPORTED_FILE_EXTENSIONS)

def get_file_extension(key: str) -> str:
    """Get file extension from S3 key"""
    return key.split('.')[-1].lower() if '.' in key else 'unknown'

def read_s3_file(bucket: str, key: str, record_id: str) -> str:
    """
    Read file content from S3 with robust error handling.
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        record_id: Record identifier for logging
        
    Returns:
        str: File content as UTF-8 string
    """
    logger.info(f"[{record_id}] Reading file from S3: s3://{bucket}/{key}")
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content_bytes = response['Body'].read()
        
        # Get file metadata
        content_type = response.get('ContentType', 'unknown')
        last_modified = response.get('LastModified', 'unknown')
        
        logger.info(f"[{record_id}] File metadata - Type: {content_type}, Modified: {last_modified}")
        
        # Try multiple encoding strategies
        content = decode_content(content_bytes, record_id)
        
        logger.info(f"[{record_id}] Successfully read {len(content)} characters")
        return content
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise ProcessingError(f"File not found: s3://{bucket}/{key}")
        elif error_code == 'AccessDenied':
            raise ProcessingError(f"Access denied to file: s3://{bucket}/{key}")
        else:
            raise ProcessingError(f"S3 error ({error_code}): {str(e)}")
    except Exception as e:
        raise ProcessingError(f"Error reading file {key} from bucket {bucket}: {str(e)}")

def decode_content(content_bytes: bytes, record_id: str) -> str:
    """
    Decode content bytes to string using multiple encoding strategies.
    
    Args:
        content_bytes: Raw file content
        record_id: Record identifier for logging
        
    Returns:
        str: Decoded content
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            content = content_bytes.decode(encoding)
            logger.info(f"[{record_id}] Successfully decoded using {encoding}")
            return content
        except UnicodeDecodeError:
            continue
    
    # Last resort: decode with error handling
    logger.warning(f"[{record_id}] Using fallback decoding with error replacement")
    return content_bytes.decode('utf-8', errors='replace')

def prepare_content_for_analysis(content: str, key: str, record_id: str) -> str:
    """
    Prepare content for Comprehend analysis.
    
    Args:
        content: Raw file content
        key: File key for logging
        record_id: Record identifier for logging
        
    Returns:
        str: Processed content ready for analysis
    """
    logger.info(f"[{record_id}] Preparing content for analysis")
    
    # Remove null characters and other problematic characters
    content = content.replace('\x00', '').replace('\ufeff', '')  # Remove BOM
    
    # Normalize whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Validate content
    if not content:
        raise FileValidationError("File contains no readable text after processing")
    
    # Handle JSON files specially
    if key.lower().endswith('.json'):
        content = extract_text_from_json(content, record_id)
    
    # Truncate if too long for Comprehend
    original_length = len(content)
    if len(content.encode('utf-8')) > MAX_COMPREHEND_TEXT_LENGTH:
        # Truncate at word boundaries when possible
        content = truncate_at_word_boundary(content, TRUNCATE_LENGTH)
        logger.warning(f"[{record_id}] Content truncated from {original_length} to {len(content)} characters")
    
    logger.info(f"[{record_id}] Content prepared: {len(content)} characters")
    return content

def extract_text_from_json(json_content: str, record_id: str) -> str:
    """Extract readable text from JSON content"""
    try:
        data = json.loads(json_content)
        text_parts = []
        
        def extract_text_recursive(obj, depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            
            if isinstance(obj, str):
                # Only include strings that look like meaningful text
                if len(obj) > 2 and not obj.isdigit():
                    text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item, depth + 1)
        
        extract_text_recursive(data)
        result = ' '.join(text_parts)
        
        if not result.strip():
            # Fallback to original JSON if no text extracted
            return json_content
        
        logger.info(f"[{record_id}] Extracted {len(result)} characters from JSON")
        return result
        
    except json.JSONDecodeError:
        logger.warning(f"[{record_id}] Invalid JSON, using raw content")
        return json_content

def truncate_at_word_boundary(text: str, max_length: int) -> str:
    """Truncate text at word boundary"""
    if len(text) <= max_length:
        return text
    
    # Find last space before max_length
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If space is reasonably close to end
        return truncated[:last_space]
    else:
        return truncated  # Just truncate at character boundary

def analyze_sentiment(text: str, record_id: str) -> Dict[str, Any]:
    """
    Analyze sentiment using Amazon Comprehend.
    
    Args:
        text: Text content to analyze
        record_id: Record identifier for logging
        
    Returns:
        Dict with sentiment analysis results
    """
    logger.info(f"[{record_id}] Starting sentiment analysis")
    
    try:
        response = comprehend_client.detect_sentiment(
            Text=text,
            LanguageCode='en'
        )
        
        sentiment = response['Sentiment']
        scores = response['SentimentScore']
        
        logger.info(f"[{record_id}] Sentiment analysis completed: {sentiment}")
        logger.debug(f"[{record_id}] Sentiment scores: {scores}")
        
        return {
            'Sentiment': sentiment,
            'SentimentScore': {
                'Positive': round(scores['Positive'], 4),
                'Negative': round(scores['Negative'], 4),
                'Neutral': round(scores['Neutral'], 4),
                'Mixed': round(scores['Mixed'], 4)
            }
        }
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"[{record_id}] Comprehend sentiment error ({error_code}): {str(e)}")
        
        # Return neutral sentiment for recoverable errors
        return get_neutral_sentiment()
        
    except Exception as e:
        logger.error(f"[{record_id}] Unexpected sentiment analysis error: {str(e)}")
        return get_neutral_sentiment()

def extract_entities(text: str, record_id: str) -> List[Dict[str, Any]]:
    """
    Extract entities using Amazon Comprehend.
    
    Args:
        text: Text content to analyze
        record_id: Record identifier for logging
        
    Returns:
        List of extracted entities
    """
    logger.info(f"[{record_id}] Starting entity extraction")
    
    try:
        response = comprehend_client.detect_entities(
            Text=text,
            LanguageCode='en'
        )
        
        entities = response['Entities']
        
        # Sort by confidence score and filter
        filtered_entities = []
        for entity in entities:
            if entity['Score'] >= 0.5:  # Only high-confidence entities
                filtered_entities.append({
                    'Text': entity['Text'][:100],  # Limit text length
                    'Type': entity['Type'],
                    'Score': round(entity['Score'], 4),
                    'BeginOffset': entity.get('BeginOffset', 0),
                    'EndOffset': entity.get('EndOffset', 0)
                })
        
        # Sort by score and take top 10
        filtered_entities.sort(key=lambda x: x['Score'], reverse=True)
        result = filtered_entities[:10]
        
        logger.info(f"[{record_id}] Entity extraction completed: {len(result)} entities found")
        return result
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"[{record_id}] Comprehend entities error ({error_code}): {str(e)}")
        return []
        
    except Exception as e:
        logger.error(f"[{record_id}] Unexpected entity extraction error: {str(e)}")
        return []

def detect_language(text: str, record_id: str) -> Dict[str, Any]:
    """
    Detect language using Amazon Comprehend.
    
    Args:
        text: Text content to analyze
        record_id: Record identifier for logging
        
    Returns:
        Dict with language detection results
    """
    logger.info(f"[{record_id}] Starting language detection")
    
    try:
        # Use shorter text sample for language detection
        sample_text = text[:1000] if len(text) > 1000 else text
        
        response = comprehend_client.detect_dominant_language(Text=sample_text)
        
        languages = response['Languages']
        if languages:
            dominant_language = languages[0]
            result = {
                'LanguageCode': dominant_language['LanguageCode'],
                'Score': round(dominant_language['Score'], 4)
            }
            logger.info(f"[{record_id}] Language detected: {result['LanguageCode']} ({result['Score']})")
            return result
        else:
            logger.warning(f"[{record_id}] No language detected")
            return {'LanguageCode': 'unknown', 'Score': 0.0}
            
    except Exception as e:
        logger.error(f"[{record_id}] Language detection error: {str(e)}")
        return {'LanguageCode': 'en', 'Score': 0.0}  # Default to English

def get_neutral_sentiment() -> Dict[str, Any]:
    """Return neutral sentiment as fallback"""
    return {
        'Sentiment': 'NEUTRAL',
        'SentimentScore': {
            'Positive': 0.0,
            'Negative': 0.0,
            'Neutral': 1.0,
            'Mixed': 0.0
        }
    }

def create_metadata(file_id: str, key: str, size: int, bucket: str, content: str,
                   sentiment_result: Dict, entities_result: List, language_result: Dict,
                   record_id: str) -> Dict[str, Any]:
    """Create comprehensive metadata object"""
    
    now = datetime.now(timezone.utc)
    
    metadata = {
        'fileId': file_id,
        'fileName': key,
        'filePath': f"s3://{bucket}/{key}",
        'fileSize': size,
        'fileType': get_file_extension(key),
        'contentLength': len(content),
        'uploadTime': now.isoformat(),
        'processingTime': now.isoformat(),
        'sentimentResult': sentiment_result['Sentiment'],
        'sentimentScore': sentiment_result['SentimentScore'],
        'entities': entities_result[:5],  # Store top 5 entities
        'entitiesCount': len(entities_result),
        'language': language_result,
        'bucket': bucket,
        'projectName': PROJECT_NAME,
        'environment': ENVIRONMENT,
        'version': '1.0.0',
        'ttl': int((now.timestamp() + (365 * 24 * 60 * 60)))  # 1 year TTL
    }
    
    logger.info(f"[{record_id}] Metadata created for file: {key}")
    return metadata

def store_metadata(metadata: Dict[str, Any], record_id: str) -> None:
    """
    Store file metadata in DynamoDB with error handling.
    
    Args:
        metadata: Metadata dictionary to store
        record_id: Record identifier for logging
    """
    logger.info(f"[{record_id}] Storing metadata in DynamoDB")
    
    try:
        # Convert float values to Decimal for DynamoDB compatibility
        processed_metadata = convert_floats_to_decimal(metadata)
        
        # Store with conditional check to prevent duplicates
        table.put_item(
            Item=processed_metadata,
            ConditionExpression='attribute_not_exists(fileId)'
        )
        
        logger.info(f"[{record_id}] Metadata stored successfully for file: {metadata['fileName']}")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ConditionalCheckFailedException':
            logger.warning(f"[{record_id}] File already processed: {metadata['fileId']}")
            # Update existing record instead
            update_existing_metadata(metadata, record_id)
        else:
            logger.error(f"[{record_id}] DynamoDB error ({error_code}): {str(e)}")
            raise DynamoDBError(f"Failed to store metadata: {str(e)}")
    except Exception as e:
        logger.error(f"[{record_id}] Unexpected error storing metadata: {str(e)}")
        raise DynamoDBError(f"Failed to store metadata: {str(e)}")

def update_existing_metadata(metadata: Dict[str, Any], record_id: str) -> None:
    """Update existing metadata record"""
    try:
        processed_metadata = convert_floats_to_decimal(metadata)
        
        table.put_item(Item=processed_metadata)
        logger.info(f"[{record_id}] Metadata updated for file: {metadata['fileName']}")
        
    except Exception as e:
        logger.error(f"[{record_id}] Failed to update metadata: {str(e)}")
        raise DynamoDBError(f"Failed to update metadata: {str(e)}")

def convert_floats_to_decimal(obj: Any) -> Any:
    """Convert float values to Decimal for DynamoDB"""
    if isinstance(obj, dict):
        return {k: convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    elif isinstance(obj, float):
        return Decimal(str(round(obj, 6)))
    else:
        return obj

def send_notification(metadata: Dict[str, Any], record_id: str) -> None:
    """
    Send processing notification via SNS.
    
    Args:
        metadata: File metadata
        record_id: Record identifier for logging
    """
    logger.info(f"[{record_id}] Sending notification")
    
    try:
        # Prepare notification content
        sentiment = metadata['sentimentResult']
        confidence = max([float(v) for v in metadata['sentimentScore'].values()])
        
        subject = f"✅ Text Processing Complete - {metadata['fileName']}"
        
        message = format_notification_message(metadata, sentiment, confidence)
        
        # Send notification
        response = sns_client.publish(
            TopicArn=SNS_TOPIC,
            Message=message,
            Subject=subject
        )
        
        message_id = response.get('MessageId', 'unknown')
        logger.info(f"[{record_id}] Notification sent successfully (MessageId: {message_id})")
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.error(f"[{record_id}] SNS error ({error_code}): {str(e)}")
        # Don't raise exception to avoid failing the entire process
    except Exception as e:
        logger.error(f"[{record_id}] Unexpected notification error: {str(e)}")
        # Don't raise exception to avoid failing the entire process

def format_notification_message(metadata: Dict[str, Any], sentiment: str, confidence: float) -> str:
    """Format the notification message"""
    
    entities_text = format_entities_for_notification(metadata['entities'])
    language_info = metadata.get('language', {})
    
    message = f"""
🔍 TEXT PROCESSING SUMMARY
{'=' * 50}

📄 FILE INFORMATION:
• File: {metadata['fileName']}
• Size: {metadata['fileSize']:,} bytes
• Type: {metadata['fileType'].upper()}
• Content Length: {metadata['contentLength']:,} characters
• Upload Time: {metadata['uploadTime']}
• Processing Time: {metadata['processingTime']}

🧠 AI ANALYSIS RESULTS:
• Sentiment: {sentiment} ({confidence:.1%} confidence)
• Language: {language_info.get('LanguageCode', 'unknown').upper()} ({language_info.get('Score', 0):.1%} confidence)
• Entities Found: {metadata['entitiesCount']}

📊 DETAILED SENTIMENT SCORES:
• Positive: {metadata['sentimentScore']['Positive']:.3f}
• Negative: {metadata['sentimentScore']['Negative']:.3f}
• Neutral: {metadata['sentimentScore']['Neutral']:.3f}
• Mixed: {metadata['sentimentScore']['Mixed']:.3f}

🏷️  TOP ENTITIES DETECTED:
{entities_text}

📍 TECHNICAL DETAILS:
• File ID: {metadata['fileId']}
• S3 Location: {metadata['filePath']}
• Project: {metadata['projectName']}
• Environment: {metadata['environment']}
• Processing Version: {metadata['version']}

💡 Need help? Check the AWS Console for detailed logs and metrics.
"""
    
    return message

def format_entities_for_notification(entities: List[Dict]) -> str:
    """Format entities list for notification message"""
    if not entities:
        return "   No high-confidence entities detected"
    
    formatted = []
    for i, entity in enumerate(entities[:5], 1):
        confidence_bar = "█" * int(entity['Score'] * 10)
        formatted.append(
            f"   {i}. {entity['Text']} ({entity['Type']}) "
            f"[{confidence_bar}] {entity['Score']:.3f}"
        )
    
    return "\n".join(formatted)

# Custom exceptions
class ProcessingError(Exception):
    """Custom exception for processing errors"""
    pass

class FileValidationError(ProcessingError):
    """Custom exception for file validation errors"""
    pass

class ComprehendError(ProcessingError):
    """Custom exception for Comprehend service errors"""
    pass

class DynamoDBError(ProcessingError):
    """Custom exception for DynamoDB errors"""
    pass

class SNSError(ProcessingError):
    """Custom exception for SNS errors"""
    pass

# Health check and utility functions
def validate_environment() -> None:
    """Validate Lambda environment configuration"""
    logger.info("Validating Lambda environment...")
    
    # Check AWS service connectivity
    try:
        # Test DynamoDB
        table.table_status
        logger.info("✅ DynamoDB connection verified")
        
        # Test SNS
        sns_client.get_topic_attributes(TopicArn=SNS_TOPIC)
        logger.info("✅ SNS connection verified")
        
        # Test Comprehend
        comprehend_client.detect_sentiment(Text="test", LanguageCode="en")
        logger.info("✅ Comprehend connection verified")
        
    except Exception as e:
        logger.error(f"❌ Environment validation failed: {str(e)}")
        raise

# Initialize validation on import
try:
    validate_environment()
except Exception as e:
    logger.warning(f"Environment validation failed (continuing anyway): {str(e)}")

logger.info("Lambda function initialized successfully")
logger.info(f"Project: {PROJECT_NAME}")
logger.info(f"Environment: {ENVIRONMENT}")
logger.info(f"Supported file types: {SUPPORTED_FILE_EXTENSIONS}")
logger.info(f"Max file size: {MAX_FILE_SIZE:,} bytes")