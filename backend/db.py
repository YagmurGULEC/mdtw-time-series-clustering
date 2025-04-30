import boto3
import os
from boto3.dynamodb.conditions import Key
import aioboto3
from typing import List, Dict, Optional

TABLE_NAME="PersonDiet"
async def list_local_tables():
    session = aioboto3.Session()
    async with session.client(
        'dynamodb',
        region_name='us-west-2',
        endpoint_url='http://localhost:8000',
        aws_access_key_id='fake',
        aws_secret_access_key='fake'
    ) as client:
        response = await client.list_tables()
        return response.get("TableNames", [])





def create_table_if_not_exists(table_name, local=False):
    """Create a DynamoDB table if it does not exist."""
    dynamodb = get_dynamodb_resource(local)
    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {
                    'AttributeName': 'person_id',
                    'KeyType': 'HASH'  # Partition key
                }
                
            ],
            AttributeDefinitions=[
                {
                    'AttributeName': 'person_id',
                    'AttributeType': 'S'
                }
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        print(f"Table {table_name} created.")
    except dynamodb.meta.client.exceptions.ResourceInUseException:
        print(f"Table {table_name} already exists.")
        table = dynamodb.Table(table_name)
    return table

def insert_record(table, record):
    """Insert a record into the DynamoDB table."""
    try:
        table.put_item(Item=record)
        print(f"Record for {record['person_id']} inserted.")
    except Exception as e:
        print(f"Error inserting record: {e}")



async def fetch_all_dynamodb_items(
    table_name: str,
    region: str = "us-west-2",
    endpoint_url: Optional[str] = "http://localhost:8000",
    aws_access_key_id: str = "fake",
    aws_secret_access_key: str = "fake"
) -> List[Dict]:
    """Fetches all items from a DynamoDB table with pagination support."""
    session = aioboto3.Session()
    all_items = []

    async with session.resource(
        "dynamodb",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    ) as dynamodb:

        table = await dynamodb.Table(table_name)

        async def scan_recursive(last_key=None):
            if last_key:
                response = await table.scan(ExclusiveStartKey=last_key)
            else:
                response = await table.scan()
            
            all_items.extend(response.get("Items", []))
            if "LastEvaluatedKey" in response:
                await scan_recursive(response["LastEvaluatedKey"])

        await scan_recursive()

    return all_items


async def delete_all_items_from_table(table_name: str):
    session = aioboto3.Session()

    async with session.resource(
        "dynamodb",
        region_name="us-west-2",
        endpoint_url="http://localhost:8000",
        aws_access_key_id="fake",
        aws_secret_access_key="fake"
    ) as dynamodb:

        table = await dynamodb.Table(table_name)

        print(f"🔍 Scanning items from {table_name} to delete...")
        async def scan_and_delete(last_key=None):
            if last_key:
                response = await table.scan(ExclusiveStartKey=last_key)
            else:
                response = await table.scan()

            async with table.batch_writer() as batch:
                for item in response["Items"]:
                    key = {"person_id": item["person_id"]}
                    await batch.delete_item(Key=key)

            if "LastEvaluatedKey" in response:
                await scan_and_delete(response["LastEvaluatedKey"])

        await scan_and_delete()
        print(f"✅ All items deleted from table '{table_name}'.")