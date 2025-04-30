from pyspark.sql import SparkSession
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("DTWExample").getOrCreate()
# Define a function to calculate DTW distance

def scalar_to_vector_euclidean(a, b):
    return euclidean([a], [b])  # wrap scalars into lists
data = [
    (f"person_{i}", [{"time": np.random.rand(3).tolist()}]) for i in range(10_000)
   
   
    
]

rdd = spark.sparkContext.parallelize(data)
dist_matrix = np.zeros((len(data), len(data)))
def calculate_dtw(pair):
    id1, series1 = pair[0]
    id2, series2 = pair[1]
    id1= int(id1.split("_")[1])
    id2= int(id2.split("_")[1])
    # Extract the time series data
    series1 = np.array(series1[0]['time'])
    series2 = np.array(series2[0]['time'])
    
    # Calculate DTW distance
    distance, _ = fastdtw(series1, series2, dist=scalar_to_vector_euclidean)

    
    return (id1, id2, distance)

# pairs
pairs = rdd.cartesian(rdd).filter(lambda x: x[0][0] < x[1][0])

results = pairs.map(calculate_dtw).collect()
for id1, id2, distance in results:
    # print(f"DTW distance between {id1} and {id2}: {distance}")
    dist_matrix[id1][id2] = distance
    dist_matrix[id2][id1] = distance
# print("Distance Matrix:")
# print(dist_matrix)

spark.stop()
