from data.utils.modified_mdtw  import local_distance
import pytest
import numpy as np



#No time penalty but different nutrient
@pytest.mark.parametrize("pair1, pair2", [
    (
        (8.0, [0.5]),
        (8.0, [0.3]),
    ),
    
    (
        (10.0, [0.2, 0.8]),
        (10.0, [0.4, 0.6]),
    ),
    


])
def test_local_distance_same_time_different_nutrient(pair1, pair2):
    v1= np.array(pair1[1])
    v2= np.array(pair2[1])
    W=np.eye(len(v1))
    res1=(v1-v2).T @ W @ (v1-v2)+2 * (v1.T @ W @ v2)*(0)**2
    res2=local_distance(pair1, pair2)
    assert res1==res2, f"Expected {res1}, but got {res2}"


#If it gives the error for different size of nutrient
@pytest.mark.parametrize("pair1, pair2", [
    (
        (8.0, [0.5]),
        (None, []),
    ),
    
    (
        (10.0, [0.2, 0.8]),
        (None, []),
    ),
      (
        
        (None, []),
        (8.0, [0.5]),
    ),
    ]
)
def test_local_distance_skipped_meals(pair1, pair2):
    with pytest.raises(ValueError):
        local_distance(pair1, pair2)

    
    