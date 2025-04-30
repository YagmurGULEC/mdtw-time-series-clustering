import numpy as np
from itertools import zip_longest
from .plotting_tools import create_time_series_plot, plot_heatmap
def mdtw_distance(ER1, ER2, traditional=False,delta=23, beta=1, alpha=2):
    m1 = len(ER1)
    m2 = len(ER2)

    # Local distance matrix including matching with empt`y
    deo = np.zeros((m1 + 1, m2 + 1))

    for i in range(m1 + 1):
        for j in range(m2 + 1):
            if i == 0 and j == 0:
                deo[i, j] = 0
            elif i == 0:
                tj, vj = ER2[j-1]
                deo[i, j] = np.dot(vj, vj)  # cost of matching to empty
            elif j == 0:
                ti, vi = ER1[i-1]
                deo[i, j] = np.dot(vi, vi)
            else:
                deo[i, j] = local_distance(ER1[i-1], ER2[j-1], traditional,delta, beta, alpha)

    # Global cost matrix
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

    return dER[m1, m2]



def local_distance(eo_i, eo_j,traditional=False,delta=23, beta=1, alpha=2):
    # eo_i and eo_j: (time, vector) tuples
   
    ti, vi = eo_i
    tj, vj = eo_j
   
    vi = np.array(vi)
    vj = np.array(vj)

    if vi.shape != vj.shape:
        if len(vi) == 0:
            return np.sum(vj ** 2)
        elif len(vj) == 0:
            return np.sum(vi ** 2)
        else:
            raise ValueError("Mismatch in feature dimensions.")
    if np.any(vi < 0) or np.any(vj < 0):
        raise ValueError("Nutrient values must be non-negative.")
    if np.any(vi>1 ) or np.any(vj>1):
        raise ValueError("Nutrient values must be in the range [0, 1].")   
    W = np.eye(len(vi))  # Assume W = identity for now
    value_diff = (vi - vj).T @ W @ (vi - vj)
    if traditional:
        return value_diff
    
    time_diff = (np.abs(ti - tj) / delta) ** alpha
    scale = 2 * beta * (vi.T @ W @ vj)
    
    distance = value_diff + scale * time_diff
  
    return distance



def generate_synthetic_data(num_people=5, min_meals=1, max_meals=5,min_calories=200,max_calories=800):
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



def calculate_distance_matrix(prepared_data,traditional=False):
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
                # zip_longest over (time, nutrients) pairs
                ER1 = list(records1.items())
                ER2 = list(records2.items())
                aligned_ER1 = []
                aligned_ER2 = []
                for pair1, pair2 in zip_longest(ER1, ER2, fillvalue=(None, [])):
                    time1, nut1 = pair1
                    time2, nut2 = pair2
                    aligned_ER1.append((time1, nut1))
                    aligned_ER2.append((time2, nut2))
           
                distance = mdtw_distance(aligned_ER1, aligned_ER2, traditional)
                
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
                
    return distance_matrix

if __name__ == "__main__":
    #Raw data from the database no guarantee of the order of the records according to time
    data= generate_synthetic_data(num_people=5, min_meals=1, max_meals=5,min_calories=200,max_calories=800)
    prepared_data = {person['person_id']: prepare_person(person) for person in data}
    print(prepared_data)
    # distance_matrix = calculate_distance_matrix(prepared_data,traditional=False)
    # print("Distance Matrix:")
    # print(distance_matrix)