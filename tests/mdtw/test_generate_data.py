import pytest 
import numpy as np
from data.utils.modified_mdtw import generate_synthetic_data

@pytest.mark.parametrize("num_samples, min_meals, max_meals,min_cal,max_cal", [ 
    (5, 1,10,100,800),  # 5 samples, 4 events, 3 features
   
])  
def test_generate_synthetic_data(num_samples,min_meals, max_meals,min_cal,max_cal):
    """
    Test the generate_synthetic_data function.
    """
    data = generate_synthetic_data(num_people=5, min_meals=min_meals, max_meals=max_meals,min_calories=min_cal,max_calories=max_cal)
    assert len(data) == num_samples, f"Expected {num_samples} samples, but got {len(data)}"
    for person in data:
        assert 'person_id' in person, "Missing 'person_id' in generated data"
        assert 'records' in person, "Missing 'records' in generated data"
        assert len(person['records']) >= min_meals, f"Expected at least {min_meals} meals, but got {len(person['records'])}"
        assert len(person['records']) <= max_meals, f"Expected at most {max_meals} meals, but got {len(person['records'])}"
        for record in person['records']:
            assert 'time' in record, "Missing 'time' in record"
            assert 'nutrients' in record, "Missing 'nutrients' in record"
            assert len(record['nutrients']) == 1, "Expected 1 nutrient per meal"