"""
S3 HANDLER MODULE

Handles reading/writing files from AWS S3.
Supports image and CSV file operations.
"""

import os
import io
import cv2
import numpy as np
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3Handler:
    """
    Handler for S3 file operations.
    """
    
    def __init__(self, bucket_name=None, region_name='us-east-1'):
        """
        Initialize S3 handler.
        
        Args:
            bucket_name: S3 bucket name (can be set later)
            region_name: AWS region
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 not installed. Install with: pip install boto3")
        
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET', 'flood-analysis')
        self.region_name = region_name
        
        try:
            self.s3_client = boto3.client('s3', region_name=region_name)
            self.s3_resource = boto3.resource('s3', region_name=region_name)
            print(f"✓ S3 connection initialized (bucket: {self.bucket_name})")
        except Exception as e:
            print(f"Error initializing S3: {e}")
            raise
    
    def read_image_from_s3(self, s3_key):
        """
        Read image from S3.
        
        Args:
            s3_key: S3 object key (path)
            
        Returns:
            np.array: Image in BGR format (OpenCV)
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_bytes = response['Body'].read()
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError(f"Failed to decode image from {s3_key}")
            
            print(f"✓ Image read from S3: s3://{self.bucket_name}/{s3_key}")
            return image
        
        except ClientError as e:
            print(f"Error reading from S3: {e}")
            raise
        except Exception as e:
            print(f"Error processing image from S3: {e}")
            raise
    
    def write_image_to_s3(self, image, s3_key):
        """
        Write image to S3.
        
        Args:
            image: Image in BGR format (OpenCV)
            s3_key: S3 object key (path)
            
        Returns:
            bool: Success
        """
        try:
            # Encode image to bytes
            _, image_bytes = cv2.imencode('.jpg', image)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=image_bytes.tobytes(),
                ContentType='image/jpeg'
            )
            
            print(f"✓ Image written to S3: s3://{self.bucket_name}/{s3_key}")
            return True
        
        except ClientError as e:
            print(f"Error writing to S3: {e}")
            raise
        except Exception as e:
            print(f"Error encoding/uploading image: {e}")
            raise
    
    def read_video_from_s3(self, s3_key, local_temp_path="/tmp/temp_video.mp4"):
        """
        Download video from S3 to local temp path.
        
        Args:
            s3_key: S3 object key
            local_temp_path: Local temporary file path
            
        Returns:
            str: Path to temporary video file
        """
        try:
            # Create temp directory if needed
            os.makedirs(os.path.dirname(local_temp_path), exist_ok=True)
            
            # Download from S3
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                local_temp_path
            )
            
            print(f"✓ Video downloaded from S3: s3://{self.bucket_name}/{s3_key}")
            return local_temp_path
        
        except ClientError as e:
            print(f"Error downloading video from S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def write_csv_to_s3(self, csv_content, s3_key):
        """
        Write CSV content to S3.
        
        Args:
            csv_content: CSV content as string or DataFrame
            s3_key: S3 object key
            
        Returns:
            bool: Success
        """
        try:
            # Convert DataFrame to string if needed
            if hasattr(csv_content, 'to_csv'):
                csv_string = csv_content.to_csv(index=False)
            else:
                csv_string = csv_content
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_string.encode('utf-8'),
                ContentType='text/csv'
            )
            
            print(f"✓ CSV written to S3: s3://{self.bucket_name}/{s3_key}")
            return True
        
        except ClientError as e:
            print(f"Error writing CSV to S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def read_csv_from_s3(self, s3_key):
        """
        Read CSV from S3.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            str: CSV content
        """
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            csv_content = response['Body'].read().decode('utf-8')
            
            print(f"✓ CSV read from S3: s3://{self.bucket_name}/{s3_key}")
            return csv_content
        
        except ClientError as e:
            print(f"Error reading CSV from S3: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise
    
    def list_images_in_s3(self, prefix="images/"):
        """
        List all images in S3 bucket under prefix.
        
        Args:
            prefix: S3 prefix to search
            
        Returns:
            list: List of S3 keys
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                return []
            
            keys = [obj['Key'] for obj in response['Contents']]
            print(f"✓ Found {len(keys)} objects in S3 under {prefix}")
            return keys
        
        except ClientError as e:
            print(f"Error listing S3 objects: {e}")
            raise
    
    def cleanup_temp_file(self, local_temp_path):
        """
        Delete temporary file.
        
        Args:
            local_temp_path: Path to temporary file
        """
        try:
            if os.path.exists(local_temp_path):
                os.remove(local_temp_path)
                print(f"✓ Temporary file cleaned up: {local_temp_path}")
        except Exception as e:
            print(f"Warning: Could not delete temp file: {e}")


if __name__ == "__main__":
    # Test S3 connection
    try:
        handler = S3Handler()
        print("S3 handler initialized successfully")
    except Exception as e:
        print(f"Error: {e}")
