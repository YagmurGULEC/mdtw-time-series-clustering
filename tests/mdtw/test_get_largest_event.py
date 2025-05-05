import pytest 
import numpy as np
from data.utils.modified_mdtw import get_largest_event

@pytest.mark.parametrize("record,expected", [
    (
       
        {0.0: [0.3], 8.0: [0.7]},
        (8.0, 0.7)
    ),
    
    (
        {1.0: [0.1], 5.0: [0.9]},
        (5.0, 0.9)
    )
])
def test_get_largest_event(record, expected):
    """
    Test the get_largest_event function.
    """
    result = get_largest_event(record)

    assert result == expected, f"Expected {expected}, but got {result}"