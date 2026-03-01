import boto3
import botocore
import os

# To use this code, you'll need to have the AWS CLI configured or set your
# access keys as environment variables.
#
# On Linux/macOS:
# export AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
# export AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"
#
# On Windows (Command Prompt):
# set AWS_ACCESS_KEY_ID="YOUR_ACCESS_KEY"
# set AWS_SECRET_ACCESS_KEY="YOUR_SECRET_KEY"

# The Boto3 client will automatically look for credentials in your environment
# variables, a shared credentials file (~/.aws/credentials), or an IAM role.


def list_s3_buckets():
    """
    Connects to AWS S3 and lists all buckets in the account.
    """
    try:
        # Create an S3 client object
        s3_client = boto3.client("s3")

        # Call the list_buckets API to get the list of buckets
        print("Listing S3 buckets...")
        response = s3_client.list_buckets()

        # Check for a valid response
        if "Buckets" in response:
            buckets = response["Buckets"]
            if buckets:
                print("Your Amazon S3 buckets are:")
                for bucket in buckets:
                    print(f"  - {bucket['Name']}")
            else:
                print("You don't have any S3 buckets in this account.")
        else:
            print("Could not retrieve a list of buckets.")

    except botocore.exceptions.NoCredentialsError:
        print("Error: AWS credentials not found.")
        print(
            "Please configure your AWS credentials or set them as environment variables."
        )
        print("The script cannot proceed without valid credentials.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    list_s3_buckets()
