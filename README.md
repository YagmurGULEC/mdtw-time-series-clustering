# mdtw-time-series-clustering
This project implements a complete data ingestion and analysis pipeline for time-series data using Modified Dynamic Time Warping (MDTW).

[![Python CI with uv](https://github.com/YagmurGULEC/mdtw-time-series-clustering/actions/workflows/tests.yml/badge.svg)](https://github.com/YagmurGULEC/mdtw-time-series-clustering/actions/workflows/tests.yml)

# What If *When* We Eat Matters As Much As *What* We Eat?

> Using Modified Dynamic Time Warping (MDTW) and clustering to uncover temporal dietary patterns

---

### 🌱 Introduction

It's well known that *what* we eat matters — but what if *when* and *how often* we eat matters just as much?

In the midst of ongoing scientific debate around the benefits of intermittent fasting, this question becomes even more intriguing. As someone passionate about machine learning and healthy living, I was inspired by a 2017 research paper \[1] exploring this intersection. The authors introduced a novel distance metric called **Modified Dynamic Time Warping (MDTW)** — a technique designed to account not only for the nutritional content of meals but also their timing throughout the day.

Motivated by their work, I built a full implementation of MDTW from scratch using Python. I applied it to cluster *simulated individuals* into temporal dietary patterns, uncovering distinct behaviors like **skippers**, **snackers**, and **night eaters**.

---

### 🔄 Why MDTW?

While MDTW may sound like a niche metric, it fills a critical gap in time-series comparison. Traditional distance measures like Euclidean distance or even classical Dynamic Time Warping (DTW) struggle with dietary data:

* People don't eat at fixed times.
* Meals vary in frequency.
* Snacking or skipping is common.

**MDTW is designed to handle this temporal misalignment**, aligning eating events by both nutrient content and timing.

---

### 📃 What You’ll See in This Article

1. **Mathematical foundation of MDTW** — explained intuitively and in LaTeX.
2. **From formula to code** — implementing MDTW in Python with dynamic programming.
3. **Generating synthetic dietary data** to simulate real-world eating behavior.
4. **Building a distance matrix** between individual eating records.
5. **Clustering individuals** with KMedoids and evaluating with silhouette and elbow methods.
6. **Visualizing clusters** as heatmaps and joint distributions.
7. **Interpreting temporal patterns** from clusters: who eats when and how much?

---

### 🎡 Conclusion: Eating Patterns as Time Series Signals

In this project, I explored how **Modified Dynamic Time Warping (MDTW)** can help uncover temporal dietary patterns — focusing not just on what we eat, but *when* and *how much*. Using **synthetic data** to simulate realistic eating behaviors, I demonstrated how MDTW can cluster individuals into distinct profiles like skippers, snackers, or night eaters.

While this experiment was based on simulated data, it lays the groundwork for applying MDTW to **real-world datasets** (e.g., NHANES) and opens up possibilities for analyzing broader behavioral trends in health, nutrition, or even beyond.

This work shows how a nuanced distance metric — designed for irregular, real-life patterns — can surface insights traditional tools may overlook. The methodology can be extended to **chrononutrition research**, **personalized health monitoring**, or any domain where **when things happen** matters just as much as **what happens**.

✨ *What we eat* is important. But *when and how* we eat might be just as crucial — and now, we have the tools to explore that.

---

### 🔗 References

\[1] Adhikari et al., *A Modified Dynamic Time Warping Distance Measure for Temporal Nutritional Pattern Analysis*, 2017

---

