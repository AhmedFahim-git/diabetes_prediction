from __future__ import annotations

import logging

import boto3
from botocore.exceptions import ClientError


def create_bucket(
    bucket_name, region="us-east-1", endpoint_url="http://localstack:4566"
):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :param region: String region to create bucket in, e.g., 'us-west-2'
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        s3_client = boto3.client("s3", region_name=region, endpoint_url=endpoint_url)
        s3_client.create_bucket(Bucket=bucket_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


if __name__ == "__main__":
    create_bucket(
        "my-bucket", region="us-east-1", endpoint_url="http://localstack:4566"
    )
