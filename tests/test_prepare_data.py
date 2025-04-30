from backend.utils.modified_mdtw  import prepare_person
import pytest
import numpy as np


# All dimensions of the nutrient records must be the same
# Test if all nutrient records have the same length



@pytest.mark.parametrize("data",[
    (
        [
            {
                'person_id': 'person_1',
                'records': [
                    {'time': 8, 'nutrients': [100]},
                    {'time': 13, 'nutrients': [200, 300]},
                    {'time': 20, 'nutrients': [100, 100]},
                ]
            },
            {
                'person_id': 'person_2',
                'records': [
                    {'time': 8, 'nutrients': [100, 100]},
                ]
            }
        ]
    )
])
def test_prepare_data_gives_error(data):
   with pytest.raises(ValueError):
        for person in data:
           prepare_person(person)

# Test if the function returns a dictionary with the correct structure
@pytest.mark.parametrize("data",[
    (
  
        [
            {
                'person_id': 'person_1',
                'records': [
                    {'time': 20, 'nutrients': [100]},
                    {'time': 8, 'nutrients': [100]},
                    {'time': 13, 'nutrients': [200]},
                    
                ]
            },
            {
                'person_id': 'person_2',
                'records': [
                    {'time': 8, 'nutrients': [100]},
                ]
            }
        ]
    )
   
])
def test_prepare_data(data):
    prepared_data = {person['person_id']: prepare_person(person) for person in data}
    
    assert isinstance(prepared_data, dict)
    assert len(prepared_data) == 2
    #check if the times are sorted
    for person_id, records in prepared_data.items():
        times = list(records.keys())
        assert times == sorted(times), f"Times for {person_id} are not sorted: {times}"
    
    assert 'person_1' in prepared_data
    assert 'person_2' in prepared_data
   
    for person_id, records in prepared_data.items():
        nutrients = np.stack(np.array([record for record in records.values()]))
    sumed_nutrients = np.sum(nutrients, axis=0)
    assert np.allclose(sumed_nutrients, 1.0), f"Sum of nutrients for {person_id} is not 1.0"
