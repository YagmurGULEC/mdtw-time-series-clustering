# mdtw-time-series-clustering
This project implements a complete data ingestion and analysis pipeline for time-series data using Modified Dynamic Time Warping (MDTW).

[![Python CI with uv](https://github.com/YagmurGULEC/mdtw-time-series-clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/YagmurGULEC/mdtw-time-series-clustering/actions/workflows/tests.yml)


## Temporal Dietary Pattern Clustering With Dynamic Time Warping

### Project Overview 
- Generating synthetic 24-hour eating event records.

- Storing the data in AWS DynamoDB.

- Calculating a Modified Dynamic Time Warping (MDTW) distance between individual dietary time series.

- Performing k-means clustering based on MDTW distance to group similar eating patterns.

- Visualizing the clusters of temporal dietary patterns on a user-friendly web interface.

## ETL Workflow

This project follows a complete ETL pipeline structure:

1. **Extract**: Generate synthetic dietary intake time series.
2. **Transform**: Apply Modified Dynamic Time Warping (MDTW) and cluster using k-means.
3. **Load**: Store and query data from DynamoDB to enable fast access and visualization.

This makes the project a scalable and realistic prototype for health analytics pipelines.

