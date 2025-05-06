import pytest
import numpy as np

from data.utils.modified_mdtw import (mdtw_distance,mdtw_distance_optimized)


# Test data
@pytest.fixture
def sample_data():
    """Generate sample event sequences for testing."""
    ER1 = [(1, [0.2, 0.3, 0.5]), (3, [0.1, 0.4, 0.2]), (5, [0.3, 0.1, 0.4])]
    ER2 = [(2, [0.3, 0.2, 0.4]), (4, [0.2, 0.3, 0.1]), (6, [0.4, 0.5, 0.2])]
    return ER1, ER2

@pytest.fixture
def numpy_sample_data():
    """Generate sample event sequences using numpy arrays."""
    ER1 = [(1, np.array([0.2, 0.3, 0.5], dtype=np.float32)), 
           (3, np.array([0.1, 0.4, 0.2], dtype=np.float32))]
    ER2 = [(2, np.array([0.3, 0.2, 0.4], dtype=np.float32)), 
           (4, np.array([0.2, 0.3, 0.1], dtype=np.float32))]
    return ER1, ER2

@pytest.fixture
def edge_cases():
    """Generate edge case sequences."""
    # Empty sequences
    empty_seq = []
    # Single event sequences
    single_er1 = [(1, [0.2, 0.3, 0.5])]
    single_er2 = [(2, [0.3, 0.2, 0.4])]
    # Identical sequences
    identical = [(1, [0.2, 0.3, 0.5]), (3, [0.1, 0.4, 0.2])]
    
    return {
        "empty": (empty_seq, empty_seq),
        "single": (single_er1, single_er2),
        "identical": (identical, identical.copy())
    }

# Tests for correctness
def test_correctness_with_original(sample_data):
    """Test that optimized function gives same results as original."""
    ER1, ER2 = sample_data
    
    # Calculate distances using both functions
    original_dist = mdtw_distance(ER1, ER2)
    optimized_dist = mdtw_distance_optimized(ER1, ER2)
    
    # Check if results are close (allowing for floating point precision differences)
    assert np.isclose(original_dist, optimized_dist), \
        f"Original: {original_dist}, Optimized: {optimized_dist}"


def test_with_numpy_arrays(numpy_sample_data):
    """Test that function handles numpy arrays correctly."""
    ER1, ER2 = numpy_sample_data
    
    # Should not raise errors and return a valid result
    result = mdtw_distance_optimized(ER1, ER2)
    assert isinstance(result, float) or isinstance(result, np.float32) or isinstance(result, np.float64)
    assert result >= 0

# Tests for edge cases
def test_empty_sequences(edge_cases):
    """Test behavior with empty sequences."""
    empty_er1, empty_er2 = edge_cases["empty"]
    
    # Should return 0 for empty-to-empty comparison
    result = mdtw_distance_optimized(empty_er1, empty_er2)
    assert result == 0

# Tests for edge cases
def test_empty_sequences(edge_cases):
    """Test behavior with empty sequences."""
    empty_er1, empty_er2 = edge_cases["empty"]
    
    # Should return 0 for empty-to-empty comparison
    result = mdtw_distance_optimized(empty_er1, empty_er2)
    assert result == 0


def test_single_event_sequences(edge_cases):
    """Test with single event sequences."""
    single_er1, single_er2 = edge_cases["single"]
    
    # Should not raise errors and return a valid result
    result = mdtw_distance_optimized(single_er1, single_er2)
    assert isinstance(result, float) or isinstance(result, np.float32) or isinstance(result, np.float64)
    assert result >= 0


def test_identical_sequences(edge_cases):
    """Test with identical sequences."""
    identical_er1, identical_er2 = edge_cases["identical"]
    
    # Identical sequences should have a distance close to 0 for value difference
    # (time difference might still contribute to non-zero result)
    result = mdtw_distance_optimized(identical_er1, identical_er2)
    assert result >= 0
    # The result should be smaller than comparing different sequences
    different_er = [(1, [0.9, 0.9, 0.9]), (3, [0.9, 0.9, 0.9])]
    different_result = mdtw_distance_optimized(identical_er1, different_er)
    assert result < different_result

def test_identical_sequences(edge_cases):
    """Test with identical sequences."""
    identical_er1, identical_er2 = edge_cases["identical"]
    
    # Identical sequences should have a distance close to 0 for value difference
    # (time difference might still contribute to non-zero result)
    result = mdtw_distance_optimized(identical_er1, identical_er2)
    assert result >= 0
    # The result should be smaller than comparing different sequences
    different_er = [(1, [0.9, 0.9, 0.9]), (3, [0.9, 0.9, 0.9])]
    different_result = mdtw_distance_optimized(identical_er1, different_er)
    assert result < different_result

def test_different_dimensions():
    """Test with mismatched dimensions."""
    ER1 = [(1, [0.2, 0.3, 0.5]), (3, [0.1, 0.4, 0.2])]
    ER2 = [(2, [0.3, 0.2]), (4, [0.2, 0.3])]  # Different number of nutrients
    
    with pytest.raises(ValueError):
        mdtw_distance_optimized(ER1, ER2)

def test_negative_values():
    """Test with negative nutrient values."""
    ER1 = [(1, [0.2, -0.3, 0.5]), (3, [0.1, 0.4, 0.2])]
    ER2 = [(2, [0.3, 0.2, 0.4]), (4, [0.2, 0.3, 0.1])]
    
    with pytest.raises(ValueError):
        mdtw_distance_optimized(ER1, ER2)


def test_out_of_range_values():
    """Test with values outside [0,1] range."""
    ER1 = [(1, [0.2, 0.3, 1.5]), (3, [0.1, 0.4, 0.2])]
    ER2 = [(2, [0.3, 0.2, 0.4]), (4, [0.2, 0.3, 0.1])]
    
    with pytest.raises(ValueError):
        mdtw_distance_optimized(ER1, ER2)


def test_negative_values():
    """Test with negative nutrient values."""
    ER1 = [(1, [0.2, -0.3, 0.5]), (3, [0.1, 0.4, 0.2])]
    ER2 = [(2, [0.3, 0.2, 0.4]), (4, [0.2, 0.3, 0.1])]
    
    with pytest.raises(ValueError):
        mdtw_distance_optimized(ER1, ER2)

def test_negative_values():
    """Test with negative nutrient values."""
    ER1 = [(1, [0.2, -0.3, 0.5]), (3, [0.1, 0.4, 0.2])]
    ER2 = [(2, [0.3, 0.2, 0.4]), (4, [0.2, 0.3, 0.1])]
    
    with pytest.raises(ValueError):
        mdtw_distance_optimized(ER1, ER2)

# Performance comparison tests
def test_performance(sample_data, benchmark):
    """Benchmark the optimized function performance."""
    ER1, ER2 = sample_data
    
    # Create longer sequences for more meaningful benchmark
    long_ER1 = ER1 * 30  # Repeat the sequence 30 times
    long_ER2 = ER2 * 30
    
    # Benchmark the function
    result = benchmark(mdtw_distance_optimized, long_ER1, long_ER2)
    # Just check that it returns a valid result
    assert isinstance(result, float) or isinstance(result, np.float32) or isinstance(result, np.float64)
    assert result >= 0

def test_memory_usage(sample_data):
    """Test memory usage (requires memory_profiler)."""
    # This test is commented out as it requires the memory_profiler package
    # If you want to run it, uncomment and install memory_profiler

    try:
        from memory_profiler import memory_usage
        import functools
        
        ER1, ER2 = sample_data
 
        # Create longer sequences for more meaningful memory measurement
        long_ER1 = ER1 * 100
        long_ER2 = ER2 * 100
        
        # Measure memory usage of original function
        original_mem = max(memory_usage(
            functools.partial(mdtw_distance, long_ER1, long_ER2)
        ))
        
        # Measure memory usage of optimized function
        optimized_mem = max(memory_usage(
            functools.partial(mdtw_distance_optimized, long_ER1, long_ER2)
        ))
        
        print(f"Original memory usage: {original_mem} MB")
        print(f"Optimized memory usage: {optimized_mem} MB")
        
        # Check that optimized version uses less memory
        assert optimized_mem < original_mem
    except ImportError:
        pytest.skip("memory_profiler not installed")

    # # pass  # Placeholder for memory test
