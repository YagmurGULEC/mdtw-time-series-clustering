
from typing import List, Dict
import asyncio
import os
import numpy as np
import aiofiles
import json
from decimal import Decimal

# Utility to convert float -> Decimal (required by DynamoDB)
def convert_floats_to_decimals(obj):
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    else:
        return obj


class AsyncJSONLWriter:
    def __init__(self, filename: str, chunk_size: int = 100,total_records: int = 1000, queue_size:int=5):
        self.filename = filename
        self.chunk_size = chunk_size
        self.total_records = total_records
        self.queue=asyncio.Queue(maxsize=queue_size)
        self.data=[]
    def generate_chunk_vectorized(self, start: int) -> List[Dict]:
        chunk = []
        for i in range(start, start + self.chunk_size):
            person_id = f"person_{i}"
            mp = np.random.randint(1, 11)
            # Unique + sorted time points (sample without replacement)
            times = np.sort(np.random.choice(range(24), mp, replace=False))
            nutrients = np.random.uniform(0, 100, size=(3, mp))

            record = {
                "person_id": person_id,
                "records": [
                    {"time": float(times[j]), "nutrients": nutrients[:, j].tolist()}
                    for j in range(mp)
                ]
            }
            chunk.append(record)
        return chunk
    

    async def producer(self):
        """Generates records in chunks and puts them in the queue."""
        start=0
        loop=asyncio.get_running_loop()
        for start in range(0, self.total_records, self.chunk_size):
    
            chunk = await loop.run_in_executor(None, self.generate_chunk_vectorized, start)
            await self.queue.put(chunk)
        await self.queue.put(None)  # Signal end of queue

    async def consumer(self):
    
        """Consumes chunks from the queue and writes them to the file."""
        async with aiofiles.open(self.filename, "w") as f:
            while True:
                chunk = await self.queue.get()
               
                if chunk is None:
                    break
                batch = "\n".join(json.dumps(r) for r in chunk) + "\n"
                await f.write(batch)


    async def run(self):
        """Run the producer and consumer tasks."""
        try:
            # Delete file if it exists
            if os.path.exists(self.filename):
                os.unlink(self.filename)
                
            # Start producer and consumer
            await asyncio.gather(
                self.producer(),
                self.consumer()
            )
        except Exception as e:
            print(f"Error during execution: {e}")

async def write_syntetic_data(file_name:str,chunk_size:int,total_records:int):
    writer=AsyncJSONLWriter(file_name,chunk_size,total_records)
    await writer.run()

if __name__=="__main__":
    asyncio.run(write_syntetic_data("./output.jsonl",10,50))
