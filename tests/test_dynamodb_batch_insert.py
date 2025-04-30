import pytest
import asyncio
import numpy as np
import tempfile
import os
import aiofiles
import json
import aioboto3
from backend.utils.eating_event_record import AsyncJSONLWriter,convert_floats_to_decimals
from backend.utils.data_generate import stream_jsonl_chunks,insert_to_dynamodb
from backend.db import fetch_all_dynamodb_items,delete_all_items_from_table


@pytest.mark.asyncio
async def test_dynamodb_batch_insert():
    # Temp JSONL
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        temp_path = tmp.name

    # Generate mock data
    try:
        writer = AsyncJSONLWriter(filename=temp_path, chunk_size=10, total_records=100)
        await writer.run()

        # Setup DynamoDB local
        session = aioboto3.Session()
        async with session.resource(
            'dynamodb',
            region_name='us-west-2',
            endpoint_url='http://localhost:8000',
            aws_access_key_id='fake',
            aws_secret_access_key='fake'
        ) as dynamodb:

            table_name = "TestTable"
            try:
                await dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[{"AttributeName": "person_id", "KeyType": "HASH"}],
                    AttributeDefinitions=[{"AttributeName": "person_id", "AttributeType": "S"}],
                    ProvisionedThroughput={"ReadCapacityUnits": 5, "WriteCapacityUnits": 5}
                )
                await asyncio.sleep(2)  # Wait for table to become active
            except dynamodb.meta.client.exceptions.ResourceInUseException:
                pass  # Table already exists

            table = await dynamodb.Table(table_name)

            # Insert from file
            await insert_to_dynamodb(table, input_path=temp_path)
        
            items=await fetch_all_dynamodb_items(table_name)
            assert (len(items)==100)
            for item in items:
                assert (len(item['records'])>=1) 
                assert (len(item['records'])<=10)
                for record in item['records']:
                    assert (len(record['nutrients'])==3)
    finally:
        os.remove(temp_path)

