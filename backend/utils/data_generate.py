import numpy as np
import pandas as pd
from typing import List, Dict,AsyncGenerator
import asyncio 
import aiofiles
import json
import os
import time 
import functools
import pandas as pd
import aioboto3
from backend.db import TABLE_NAME,fetch_all_dynamodb_items,delete_all_items_from_table
from .eating_event_record import AsyncJSONLWriter,convert_floats_to_decimals


async def stream_jsonl_chunks(file_path:str,chunk_size:int=25)->AsyncGenerator[List[dict], None]:
    chunk=[]
    async with aiofiles.open(file_path,mode='r') as f:
        async for line in f:
            record=json.loads(line)
            chunk.append(record)
            if len(chunk)==chunk_size:
                yield chunk
                chunk=[]
        if chunk:
            yield chunk


async def insert_to_dynamodb(table,input_path:str,chunk_size:int=25)->None:
    async for chunk in stream_jsonl_chunks(input_path, chunk_size=chunk_size):
        async with table.batch_writer() as batch:
            for item in chunk:
                item = convert_floats_to_decimals(item)  # 🔁 Convert before insert
                await batch.put_item(Item=item)  # ✅ Fully async

async def main():
    input_path=os.path.join(".","output.jsonl")
    
    json_generator = stream_jsonl_chunks(input_path)
    
    session = aioboto3.Session()
    async with session.resource(
        'dynamodb',
        region_name='us-west-2',
        endpoint_url='http://localhost:8000',
        aws_access_key_id='fake',
        aws_secret_access_key='fake'
    ) as dynamodb:

        table = await dynamodb.Table(TABLE_NAME) 
        items= await fetch_all_dynamodb_items(table_name=TABLE_NAME)
        print(f"Fetched {len(items)} items from the database.")
        
if __name__ == "__main__":
   asyncio.run(main())
    