import pytest
import numpy as np
from backend.utils.modified_mdtw import mdtw_distance





@pytest.mark.parametrize("ER1, ER2", [
    (
         [(8, [0.5, 0.5]), (13, [0.3, 0.7]), (20, [0.2, 0.8])],
         [(8, [0.5, 0.5]), (13, [0.3, 0.7]), (20, [0.2, 0.8])]
    ),
    (
        [(8, [0.5, 0.5]), (13, [0.3, 0.7])],
        []
    ),
    (
        [(8, [0.5, 0.5]), (13, [0.3, 0.7])],
        [(9, [0.5, 0.5]), (14, [0.3, 0.7])]
    )
   
    

])
def test_local_distance_shape_mismatch(ER1, ER2):
   distance=mdtw_distance(ER1, ER2)
   assert isinstance(distance, float)
   assert distance >= 0, "Distance should be non-negative"
   print (f"Distance between {ER1} and {ER2}: {distance}")