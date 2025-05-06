import pytest
from data.utils.modified_mdtw import  (mdtw_distance_optimized,mdtw_distance,
                                       calculate_distance_matrix, generate_synthetic_data, prepare_person)

@pytest.fixture
def sample_data():
    """Generate sample event sequences for testing."""
    data=generate_synthetic_data(num_people=10, min_meals=1, max_meals=5)
    prepared_data = {person['person_id']: prepare_person(person) for person in data}
    return prepared_data

def test_calculate_distance_matrix(sample_data):
    """Test the distance matrix calculation."""
    prepared_data = sample_data
    distance_matrix = calculate_distance_matrix(prepared_data,mdtw_distance_optimized)
    
    # Check if the distance matrix is square
    assert distance_matrix.shape[0] == distance_matrix.shape[1], "Distance matrix is not square"
    
    # Check if the diagonal is zero (distance to self)
    for i in range(distance_matrix.shape[0]):
        assert distance_matrix[i, i] == 0, f"Diagonal element {i} is not zero"
    
    # Check if the matrix is symmetric
    assert (distance_matrix == distance_matrix.T).all(), "Distance matrix is not symmetric"

def test_memory_usage():
    """Test the memory usage of the distance matrix calculation."""
    from memory_profiler import memory_usage
    import functools
    sample_data = generate_synthetic_data(num_people=100, min_meals=1, max_meals=5)
    prepared_data = {person['person_id']: prepare_person(person) for person in sample_data}
    # Warm-up run (not measured)
    _ = calculate_distance_matrix(prepared_data, mdtw_distance)
    _ = calculate_distance_matrix(prepared_data, mdtw_distance_optimized)
    mem_usage = memory_usage((calculate_distance_matrix, (prepared_data, mdtw_distance)), max_usage=True)
    mem_usage_optimized = memory_usage((calculate_distance_matrix, (prepared_data, mdtw_distance_optimized)), max_usage=True)
    print(f"Memory usage: {mem_usage} MiB")
    print(f"Memory usage optimized: {mem_usage_optimized} MiB")
    



