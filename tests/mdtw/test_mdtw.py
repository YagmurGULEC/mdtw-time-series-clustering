import pytest
import numpy as np
from data.utils.modified_mdtw import calculate_distance_matrix



@pytest.mark.parametrize("prepared_data", [
    (
        {'person_1': {0.0: [0.35], 8.0: [0.25], 16.0: [0.1], 20.0: [0.3]}, 
        'person_2': {7.0: [0.3], 10.0: [0.2], 
                     12.0: [0.3], 19.0: [0.1]},
                     'person_3': {0.0: [0.5], 5.0: [0.5]}}
    ),
  
   
])
def test_distance_matrix(prepared_data):
    """
    Test the distance matrix calculation.
    """
    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(prepared_data,traditional=True)
    assert (distance_matrix[0,0] == 0)

    print (distance_matrix)
     
    # Check the shape of the distance matrix
    assert distance_matrix.shape == (len(prepared_data), len(prepared_data)), "Distance matrix shape is incorrect."

    # Check the values in the distance matrix
    # assert np.allclose(distance_matrix[0, 0], 0), "Distance from person_1 to themselves should be 0."
    # assert np.allclose(distance_matrix[1, 1], 0), "Distance from person_2 to themselves should be 0."