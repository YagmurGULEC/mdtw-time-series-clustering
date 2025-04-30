import matplotlib.pyplot as plt
import numpy as np

def create_time_series_plot(data):
    plt.figure(figsize=(10, 5))
    for person in data:
        person_id = person['person_id']
        # Extract time and nutrient values
        times = [record['time'] for record in person['records']]
        nutrients = [np.mean(record['nutrients']) for record in person['records']]
        
        # Instead of plotting inside a loop, plot all at once
        plt.plot(times,nutrients, marker='o', label=person_id)

    plt.title('Time Series Plot for Nutrient Data')
    plt.xlabel('Time')
    plt.ylabel('Nutrient Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_series.png')
def plot_heatmap(matrix, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xticks(ticks=range(len(matrix)), labels=[f'Person {i+1}' for i in range(len(matrix))])
    plt.yticks(ticks=range(len(matrix)), labels=[f'Person {i+1}' for i in range(len(matrix))])
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('heatmap.png')