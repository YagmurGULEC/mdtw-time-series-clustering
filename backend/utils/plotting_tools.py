import matplotlib.pyplot as plt
import numpy as np

def create_time_series_plot(data,save_path=None):
    plt.figure(figsize=(10, 5))
    for person,record in data.items():
        #in case the nutrient vector has more than one dimension
        data=[[time, float(np.mean(np.array(value)))] for time,value in record.items()]

        time = [item[0] for item in data]
        nutrient_values = [item[1] for item in data]
        # Plot the time series
        plt.plot(time, nutrient_values, label=person, marker='o')

    plt.title('Time Series Plot for Nutrient Data')
    plt.xlabel('Time')
    plt.ylabel('Nutrient Value')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)

def plot_heatmap(matrix, title,save_path=None):
    """
    Plot a heatmap of the distance matrix.  
    Args:
        matrix (np.ndarray): The distance matrix.
        title (str): The title of the plot.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xticks(ticks=range(len(matrix)), labels=[f'Person {i+1}' for i in range(len(matrix))])
    plt.yticks(ticks=range(len(matrix)), labels=[f'Person {i+1}' for i in range(len(matrix))])
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
