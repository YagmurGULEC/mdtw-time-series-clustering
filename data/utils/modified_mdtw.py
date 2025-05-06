import numpy as np
from itertools import zip_longest


def mdtw_distance_optimized(ER1, ER2, delta=23, beta=1, alpha=2):
    """
    Calculate the modified DTW distance between two sequences of events
    with optimized memory usage.
    
    Args:
        ER1 (list): First sequence of events (time, nutrients).
        ER2 (list): Second sequence of events (time, nutrients).
        delta (float): Time scaling factor.
        beta (float): Weighting factor for time difference.
        alpha (float): Exponent for time difference scaling.
         
    Returns:
        float: Modified DTW distance.
    """
    m1 = len(ER1)
    m2 = len(ER2)
    
    # Only store two rows of the cost matrix at a time
    prev_row = np.zeros(m2 + 1, dtype=np.float32)
    curr_row = np.zeros(m2 + 1, dtype=np.float32)
    
    # Initialize first row (matching ER2 elements with empty)
    for j in range(1, m2 + 1):
        tj, vj = ER2[j-1]
        empty_cost = np.dot(vj, vj)  # Compute once and reuse
        prev_row[j] = prev_row[j-1] + empty_cost
    
    # Process rows one at a time
    for i in range(1, m1 + 1):
        ti, vi = ER1[i-1]
        vi_dot = np.dot(vi, vi)  # Pre-compute for reuse
        
        # First column (matching ER1 elements with empty)
        curr_row[0] = prev_row[0] + vi_dot
        
        # Fill current row
        for j in range(1, m2 + 1):
            tj, vj = ER2[j-1]
            
            # Calculate local distance only when needed
            local_dist = local_distance(
                (ti, vi), (tj, vj), delta, beta, alpha
            )
            
            # Calculate DTW options
            match_both = prev_row[j-1] + local_dist
            match_i_to_empty = prev_row[j] + vi_dot
            match_j_to_empty = curr_row[j-1] + np.dot(vj, vj)
            
            # Choose minimum path
            curr_row[j] = min(match_both, match_i_to_empty, match_j_to_empty)
        
        # Swap rows for next iteration
        prev_row, curr_row = curr_row, prev_row
    
    # Final result is in prev_row due to the last swap
    return prev_row[m2]

def mdtw_distance(ER1, ER2, delta=23, beta=1, alpha=2):
    """
    Calculate the modified DTW distance between two sequences of events.
    Args:
        ER1 (list): First sequence of events (time, nutrients).
        ER2 (list): Second sequence of events (time, nutrients).
        delta (float): Time scaling factor.
        beta (float): Weighting factor for time difference.
        alpha (float): Exponent for time difference scaling.
    
    Returns:
        float: Modified DTW distance.
    """
    m1 = len(ER1)
    m2 = len(ER2)
   
    # Local distance matrix including matching with empty
    deo = np.zeros((m1 + 1, m2 + 1))

    for i in range(m1 + 1):
        for j in range(m2 + 1):
            if i == 0 and j == 0:
                deo[i, j] = 0
            elif i == 0:
                tj, vj = ER2[j-1]
                deo[i, j] = np.dot(vj, vj)  
            elif j == 0:
                ti, vi = ER1[i-1]
                deo[i, j] = np.dot(vi, vi)
            else:
                deo[i, j]=local_distance(ER1[i-1], ER2[j-1], delta, beta, alpha)
        

    # # Global cost matrix
    dER = np.zeros((m1 + 1, m2 + 1))
    dER[0, 0] = 0

    for i in range(1, m1 + 1):
        dER[i, 0] = dER[i-1, 0] + deo[i, 0]
    for j in range(1, m2 + 1):
        dER[0, j] = dER[0, j-1] + deo[0, j]

    for i in range(1, m1 + 1):
        for j in range(1, m2 + 1):
            dER[i, j] = min(
                dER[i-1, j-1] + deo[i, j],   # Match i and j
                dER[i-1, j] + deo[i, 0],     # Match i to empty
                dER[i, j-1] + deo[0, j]      # Match j to empty
            )
   
    
    return dER[m1, m2]  # Return the final cost



def local_distance(eo_i, eo_j,delta=23, beta=1, alpha=2):
    """
    Calculate the local distance between two events.
    Args:
        eo_i (tuple): Event i (time, nutrients).
        eo_j (tuple): Event j (time, nutrients).
        delta (float): Time scaling factor.
        beta (float): Weighting factor for time difference.
        alpha (float): Exponent for time difference scaling.
    Returns:
        float: Local distance.
    """
    
   
    ti, vi = eo_i
    tj, vj = eo_j
   
    vi = np.array(vi)
    vj = np.array(vj)

    if vi.shape != vj.shape:
        raise ValueError("Mismatch in feature dimensions.")
    if np.any(vi < 0) or np.any(vj < 0):
        raise ValueError("Nutrient values must be non-negative.")
    if np.any(vi>1 ) or np.any(vj>1):
        raise ValueError("Nutrient values must be in the range [0, 1].")   
    W = np.eye(len(vi))  # Assume W = identity for now
    value_diff = (vi - vj).T @ W @ (vi - vj)
    time_diff = (np.abs(ti - tj) / delta) ** alpha
    scale = 2 * beta * (vi.T @ W @ vj)
    distance = value_diff + scale * time_diff
  
    return distance



def generate_synthetic_data(num_people=5, min_meals=1, max_meals=5,min_calories=200,max_calories=800):
    """
    Generate synthetic data for a given number of people.
    Args:
        num_people (int): Number of people to generate data for.
        min_meals (int): Minimum number of meals per person.
        max_meals (int): Maximum number of meals per person.
        min_calories (int): Minimum calories per meal.
        max_calories (int): Maximum calories per meal.
    Returns:
        list: List of dictionaries containing synthetic data for each person.
    """
    data = []
    np.random.seed(42)  # For reproducibility
    for person_id in range(1, num_people + 1):
        num_meals = np.random.randint(min_meals, max_meals + 1)  # random number of meals between min and max
        meal_times = np.sort(np.random.choice(range(24), num_meals, replace=False))  # random times sorted

        raw_calories = np.random.randint(min_calories, max_calories, size=num_meals)  # random calories between min and max
    

        person_record = {
            'person_id': f'person_{person_id}',
            'records': [
                {'time': float(time), 'nutrients': [float(cal)]} for time, cal in zip(meal_times, raw_calories)
            ]
        }

        data.append(person_record)
    return data


def prepare_person(person):
    """
    Prepare a person's data for distance calculation.
    Args:
        person (dict): Dictionary containing person's data.
    Returns:
        dict: Dictionary with time as keys and normalized nutrient vectors as values.
    """
    
    # Check if all nutrients have same length
    nutrients_lengths = [len(record['nutrients']) for record in person["records"]]
    
    if len(set(nutrients_lengths)) != 1:
        raise ValueError(f"Inconsistent nutrient vector lengths for person {person['person_id']}.")

    sorted_records = sorted(person["records"], key=lambda x: x['time'])

    nutrients = np.stack([np.array(record['nutrients']) for record in sorted_records])
    total_nutrients = np.sum(nutrients, axis=0)

    # Check to avoid division by zero
    if np.any(total_nutrients == 0):
        raise ValueError(f"Zero total nutrients for person {person['person_id']}.")

    normalized_nutrients = nutrients / total_nutrients

    # Return a dictionary {time: [normalized nutrients]}
    person_dict = {
        record['time']: normalized_nutrients[i].tolist()
        for i, record in enumerate(sorted_records)
    }

    return person_dict



def calculate_distance_matrix(prepared_data):
    """
    Calculate the distance matrix for the prepared data.
    
    Args:
        prepared_data (dict): Dictionary containing prepared data for each person.
        
    Returns:
        np.ndarray: Distance matrix.
    """
    n = len(prepared_data)
    distance_matrix = np.zeros((n, n))
    
    # Step 3: Compute pairwise distances
    for i, (id1, records1) in enumerate(prepared_data.items()):
        for j, (id2, records2) in enumerate(prepared_data.items()):
            if i < j:  # Only upper triangle
            
                ER1 = list(records1.items())
                ER2 = list(records2.items())
                
                distance_matrix[i, j] = mdtw_distance(ER1, ER2)
                distance_matrix[j, i] = distance_matrix[i, j]  # Symmetric matrix
                
    return distance_matrix

# Find the time and fraction of their largest eating occasion
def get_largest_event(record):
    """
    Find the time and fraction of the largest eating occasion in a person's record.
    Args:
        record (dict): Dictionary containing person's data.
    Returns:
        tuple: Time of the largest eating occasion and its fraction of total nutrients.
    """
    total = sum(v[0] for v in record.values())
    largest_time, largest_value = max(record.items(), key=lambda x: x[1][0])
    fractional_value = largest_value[0] / total if total > 0 else 0
    return largest_time, fractional_value

if __name__ == "__main__":
    
    prepared_data = {
        'person_1': {0.0: [0.35], 8.0: [0.25], 16.0: [0.1], 20.0: [0.3]},
        'person_2': {7.0: [0.3], 10.0: [0.2], 12.0: [0.3], 19.0: [0.1]},
        'person_3': {0.0: [0.5], 5.0: [0.5]}
    }
    distance_matrix = calculate_distance_matrix(prepared_data)
    print(distance_matrix)
