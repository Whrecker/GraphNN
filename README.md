# GNN Anomaly Detection

This repository contains a comprehensive framework for modeling and detecting structural anomalies in human daily activities using Graph Neural Networks (GNNs). 

By representing daily routines as non-Euclidean behavioral graphs, this project compares 16 distinct anomaly detection architectures, systematically pairing three state-of-the-art spatial graph convolution operators with four downstream classification mechanisms.

## Installation
To get started, install the necessary dependencies using:
```bash
pip install -r requirements.txt
```

## Dataset Generation
This project relies on a synthetic dataset representing human daily routines (e.g., sleeping, eating, bathing) captured as structural graphs.
To generate the baseline (healthy) and anomalous (different) datasets, execute the probabilistic event simulator scripts inside their respective folders:
```bash
# Generate 100 normative healthy days
py healty/reference_days.py

# Generate 100 structurally anomalous days
py different/different_days.py
```

## Evaluated Architectures (16 Models)
Every model in this repository is built using a standardized 2-layer Graph Neural Network, applying Global Mean Pooling, and training for 1000 epochs to allow embeddings to organically separate. 

We cross-evaluate **Three Graph Operators:**
- **GCN** (Graph Convolutional Networks)
- **GAT** (Graph Attention Networks)
- **GraphSAGE** (Sample and Aggregate)

Paired with **Four Classification Mechanisms:**
1. **Unsupervised Clustering (K-Means):** e.g., `clustering.py`, `gat_kmeans.py`, `graphsage_kmeans.py`
2. **Semi-Supervised Outlier Detection (One-Class SVM):** e.g., `svm_clustering.py`, `gat_svm.py`, `graphsage_svm.py`
3. **Supervised Classification (Random Forest):** e.g., `gcn_random_forest.py`, `gat_random_forest.py`, `graphsage_random_forest.py`
4. **Direct Graph Similarity Modeling (Siamese Network):** e.g., `gnn_pytorch.py`, `gcn_similarity.py`, `graphsage_similarity.py`, `attentiongnn.py` (which includes feature-attention augmentation).

## Usage
To run any of the anomaly detection pipelines, simply execute the python script of your choice:
```bash
py graphsage_kmeans.py
```

## Results
Each script automatically calculates evaluation metrics (Accuracy, Precision, Recall) on the test split. The results from any executed script will be appended sequentially to `metrics_results.xlsx` for easy visualization and comparison.
