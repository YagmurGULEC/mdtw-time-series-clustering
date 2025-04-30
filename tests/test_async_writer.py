import pytest
import asyncio
import numpy as np
import tempfile
import os
import aiofiles
import json
from backend.utils.eating_event_record import AsyncJSONLWriter

@pytest.fixture(autouse=True)
def seed():
    """Set the random seed for reproducibility."""
    np.random.seed(42)


@pytest.mark.parametrize("total_records, chunk_size", [
    (1000, 100),
    (1000, 200),
    (1000, 50),
    (500, 100),
])
@pytest.mark.asyncio
async def test_async_jsonl_writer(total_records, chunk_size):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        temp_path = tmp.name

    print(f"Temporary file: {temp_path}")
   
    writer = AsyncJSONLWriter(filename=temp_path, chunk_size=chunk_size, total_records=total_records)
    await writer.run()
    async with aiofiles.open(temp_path, "r") as f:
        lines = await f.readlines()
    assert len(lines)==total_records
     # Check JSON parseability
    for line in lines:
        obj = json.loads(line)
        assert "person_id" in obj
        assert isinstance(obj["records"], list)
    assert writer.chunk_size == chunk_size
    os.remove(temp_path)
  




