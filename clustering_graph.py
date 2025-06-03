import os
import statistics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from functools import partial
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool,GINEConv
from torch_cluster import knn_graph, graclus
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from captum.attr import IntegratedGradients
from lime.lime_tabular import LimeTabularExplainer
    
def fit_one_hot_encoder(reference_graph):
    node_names = [node[0] for node in reference_graph['nodes']]
    locations = [node[-1] for node in reference_graph['nodes']]
    node_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    location_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    node_encoder.fit(np.array(node_names).reshape(-1, 1))
    location_encoder.fit(np.array(locations).reshape(-1, 1))
    return node_encoder, location_encoder

# Create a PyG Data object from your graph (this function remains largely the same)
def create_graph_data(graph, node_encoder, location_encoder, ref_node_names):
    nodes = graph['nodes']
    edges = graph['edges']

    node_names = [node[0] for node in nodes]
    node_attributes = {node[0]: node[1:] for node in nodes}

    sample_name_encoding = node_encoder.transform([["sample"]])[0]
    sample_location_encoding = location_encoder.transform([["sample_loc"]])[0]
    name_dim = len(sample_name_encoding)
    location_dim = len(sample_location_encoding)

    node_features = []
    for name in ref_node_names:
        if name in node_attributes:
            encoded_name = node_encoder.transform([[name]])[0]  
            attributes = node_attributes[name][:-1]  
            location_str = node_attributes[name][-1]  
            encoded_location = location_encoder.transform([[location_str]])[0]  
        else:
            encoded_name = np.zeros(name_dim)
            attributes = [0] * (len(nodes[0]) - 2)
            encoded_location = np.zeros(location_dim)
        full_features = np.hstack([encoded_name, attributes, encoded_location])
        if len(full_features) != (name_dim + len(attributes) + location_dim):
            raise ValueError(f"Inconsistent feature length for node {name}")
        node_features.append(full_features)

    node_features = torch.tensor(node_features, dtype=torch.float)

    node_map = {node: i for i, node in enumerate(ref_node_names)}
    edge_index = []
    edge_weights = []
    for edge in edges:
        node1, node2, attributes = edge
        if node1 in node_map and node2 in node_map:
            idx1, idx2 = node_map[node1], node_map[node2]
            edge_index.append([idx1, idx2])
            edge_index.append([idx2, idx1])
            edge_weights.append(attributes['weights'])
            edge_weights.append(attributes['weights'])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # Return a PyG Data object; here we do not set a batch (for single graph predictions)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_weights)

def getting_data_in_dict(data):
    activity_name = []
    locations=[]
    start_time = {}
    duration = {}
    end_time = {}
    reference_graph = {"nodes": [], "edges": []}
    for i, j in data.iterrows():
        activity_name.append(j[0])
        locations.append(j[1])
        try:
            start_time[str(j[0])].append(j[2])
            duration[str(j[0])].append(j[3])
            end_time[str(j[0])].append(j[4])
        except:
            start_time[str(j[0])] = [j[2]]
            duration[str(j[0])] = [j[3]]
            end_time[str(j[0])] = [j[4]]
    for i1, i2, i3, i4,i5 in zip(activity_name, start_time, duration, end_time,locations):
        reference_graph['nodes'].append((i1, statistics.mean(start_time[i2]), statistics.mean(duration[i3]), statistics.mean(end_time[i4]),i5))
    
    
    temp = {}
    temp_list=[]
    for i1,j1 in data.iterrows():
        temp_list.append(j1[0])
        if len(temp_list)>1:
            try:
                temp[(temp_list[-2],temp_list[-1])]+=1
            except:
                temp[(temp_list[-2],temp_list[-1])]=1
    for i, j in temp.items():
        i1, i2 = i
        reference_graph['edges'].append((i1, i2, {'weights': j/10}))
    
    return reference_graph

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(2)
data_train = pd.DataFrame()
data_test = pd.DataFrame()
data_train_dif = pd.DataFrame()
data_test_dif = pd.DataFrame()
l1 = []
counter = 0
l2=[]
l3=[]
l4=[]

# Load healthy data
for i in os.listdir("C:\\Users\\jag7b\\project ankit sir\\healty"):
    if i[-4:] == "xlsx":
        counter += 1
        if counter <= 10:
            df = pd.read_excel("C:\\Users\\jag7b\\project ankit sir\\healty\\" + i)
            l4.append(df)
            data_test = pd.concat([data_test, df])    
        elif counter>10 and counter <=20:
            df = pd.read_excel("C:\\Users\\jag7b\\project ankit sir\\healty\\" + i)
            #add pd.concat for taking data for multiple days
            data_train = pd.concat([data_train,df])
        else:
            df = pd.read_excel("C:\\Users\\jag7b\\project ankit sir\\healty\\" + i)
            l2.append(df)
print("done1")
# Load different data
counter = 0
for i in os.listdir("C:\\Users\\jag7b\\project ankit sir\\different"):
    if i[-4:] == "xlsx":
        counter += 1
        if counter < 80:
            df = pd.read_excel("C:\\Users\\jag7b\\project ankit sir\\different\\" + i)
            data_test_dif = pd.concat([data_test_dif, df])
            l1.append(df)
                
        else:
            df = pd.read_excel("C:\\Users\\jag7b\\project ankit sir\\different\\" + i)
            l3.append(df)
            data_train_dif = pd.concat([data_train_dif, df])
print("done2")
# Process data

other_graph = []
for i in l1:
    i.reset_index(drop=True, inplace=True)
    other_graph.append(getting_data_in_dict(i))

data_test.reset_index(drop=True, inplace=True)
data_train.reset_index(drop=True, inplace=True)

# Create reference graph
reference_graph = getting_data_in_dict(data_train)

# Fit OneHotEncoder
encoder,encoder2 = fit_one_hot_encoder(reference_graph)
ref_node_names = [node[0] for node in reference_graph['nodes']]
ref_graph_data = create_graph_data(reference_graph, encoder,encoder2, ref_node_names)

ref_graph_data_list=[]
partial_ref_graph_data=create_graph_data({'nodes':reference_graph['nodes'][:7],'edges':reference_graph['edges'][:6]},encoder,encoder2,ref_node_names)
for i in range(2,36):
    ref_graph_data_list.append(create_graph_data({'nodes':reference_graph['nodes'][:i],'edges':reference_graph['edges'][:i-1]},encoder,encoder2,ref_node_names))
other_graph_data = []
for i in other_graph:
    other_graph_data.append(create_graph_data(i, encoder,encoder2, ref_node_names))

test1=create_graph_data(getting_data_in_dict(data_test),encoder,encoder2,ref_node_names)
test2=create_graph_data(getting_data_in_dict(data_train_dif),encoder,encoder2,ref_node_names)

partial_test1_dict=getting_data_in_dict(data_test)
partial_test2_dict=getting_data_in_dict(data_train_dif)
partial_test3_dict=getting_data_in_dict(l2[0])
partial_test4_dict=getting_data_in_dict(l3[0])
partial_test5_dict=getting_data_in_dict(l4[0])
partial_test_list1=[]
partial_test_list2=[]
partial_test_list3=[]
partial_test_list4=[]
partial_test_list5=[]
for i in range(2,36):
    partial_test_list1.append(create_graph_data({'nodes':partial_test1_dict['nodes'][:i],'edges':partial_test1_dict['edges'][:i-1]},encoder,encoder2,ref_node_names))
    partial_test_list2.append(create_graph_data({'nodes':partial_test2_dict['nodes'][:i],'edges':partial_test2_dict['edges'][:i-1]},encoder,encoder2,ref_node_names))

    partial_test_list3.append(create_graph_data({'nodes':partial_test3_dict['nodes'][:i],'edges':partial_test3_dict['edges'][:i-1]},encoder,encoder2,ref_node_names))
    partial_test_list4.append(create_graph_data({'nodes':partial_test4_dict['nodes'][:i],'edges':partial_test4_dict['edges'][:i-1]},encoder,encoder2,ref_node_names))
    
    partial_test_list5.append(create_graph_data({'nodes':partial_test5_dict['nodes'][:i],'edges':partial_test5_dict['edges'][:i-1]},encoder,encoder2,ref_node_names))
full_partial_test=[]
full_partial_test.extend(partial_test_list1)
full_partial_test.extend(partial_test_list2)
full_partial_test.extend(partial_test_list4)
full_partial_test.extend(partial_test_list5)
datas=[]
for i in l2:
    datas.append(create_graph_data(getting_data_in_dict(i),encoder,encoder2,ref_node_names))        
for i in l3:
    datas.append(create_graph_data(getting_data_in_dict(i),encoder,encoder2,ref_node_names))
for i in l4:
    datas.append(create_graph_data(getting_data_in_dict(i),encoder,encoder2,ref_node_names))
temp_count=1
training_list=[]
for i in l2:
    training_list.append(create_graph_data(getting_data_in_dict(i),encoder,encoder2,ref_node_names))
training_list.extend(partial_test_list3)
# Set dimensions
in_channels = ref_graph_data.x.size(1)     # As given: each node has 29 features.
embedding_dim = 16   # Chosen embedding dimension.

class FeatureAttention(torch.nn.Module):
    """Computes attention scores for node features."""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(input_dim, ref_graph_data.x.size(1)),  # Score per feature
            torch.nn.Sigmoid()  # Normalize scores to [0, 1]
        )
    
    def forward(self, x):
        scores = self.attention(x)  # [num_nodes, 1]
        return x * scores  # Weighted features

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.feature_attn = FeatureAttention(input_dim)
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.feature_attn(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        # Graph embedding (mean pooling)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        return global_mean_pool(x, batch), self.feature_attn.attention[0].weight.squeeze()
    
print(ref_graph_data)
print(ref_graph_data.x)
print(ref_graph_data.edge_index)
print(ref_graph_data.edge_attr.shape)    
# Initialize model and optimizer
model = GNN(input_dim=ref_graph_data.x.size(1), hidden_dim=16, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


def get_embeddings(model, graph_list):
    embeddings = []
    all_attn_scores=[]
    for graph in graph_list:
        with torch.no_grad():
            emb,attn= model(
                graph.x, 
                graph.edge_index
            )
        embeddings.append(emb.squeeze().detach().numpy())
        all_attn_scores.append(attn.squeeze().detach().numpy())
    return np.array(embeddings),np.array(all_attn_scores)
feature_importance=0
model.train()
ref_embedding,_ = model(ref_graph_data.x, ref_graph_data.edge_index)
ref_embedding = ref_embedding.detach()
for epoch in range(2000):
    total_loss = 0
    optimizer.zero_grad()
    for data in training_list:
        adj_recon,attn_scores = model(data.x, data.edge_index)
        loss = F.mse_loss(adj_recon, ref_embedding)
        loss.backward()
        total_loss += loss.item()
        feature_importance += attn_scores.detach().cpu().abs().mean(dim=0)
    optimizer.step()
    if epoch%10==0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(training_list)}")
    #if total_loss/len(training_list)<0.05:
        #break
print(feature_importance/(len(training_list) * 5000))
print(sum(feature_importance/(len(training_list) * 5000)))
embedding, _ = get_embeddings(model, datas)
print(embedding)
embeddings = embedding.mean(axis=1)
embeddings=embeddings.reshape(-1, 1)
embeddings=embeddings.astype(np.float64)
print(embeddings.shape)

k = 2  # Adjust based on your use case
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(embeddings)
acc_checker=[]
for i in range(len(datas)):
    if i<len(l2):
        acc_checker.append(0)
    if i>=len(l2) and i<len(l2)+len(l3):
        acc_checker.append(1)
    if i>=len(l2)+len(l3) and i<len(l2)+len(l3)+len(l4):
        acc_checker.append(0)
# Print cluster assignments
acc=0
for i, label in enumerate(cluster_labels):
    if acc_checker[i]==label:
        acc+=1
print(acc/len(datas))

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embedding)  # now shape (n_graphs, 2)

# 4) Plot
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(embeddings_2d):
    color = 'red' if cluster_labels[i] == 0 else 'blue'
    correct = (cluster_labels[i] == acc_checker[i])
    marker = 'o' if correct else 'x'
    plt.scatter(x, y, c=color, marker=marker, s=100,
                label=('Correct' if correct else 'Incorrect') if i == 0 else "",
                alpha=0.7)
    plt.text(x + 0.5, y + 0.5, str(i), fontsize=8, alpha=0.6)

plt.title("t-SNE Visualization of Graph Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.legend()
plt.tight_layout()


embedding, _ = get_embeddings(model, full_partial_test)
embeddings = embedding.mean(axis=1)
embeddings=embeddings.reshape(-1, 1)
embeddings=embeddings.astype(np.float64)

k = 2  # Adjust based on your use case
kmeans = KMeans(n_clusters=2, random_state=0)
cluster_labels = kmeans.fit_predict(embeddings)
acc_checker=[]
for i in range(len(full_partial_test)):
    if i<len(partial_test_list1):
        acc_checker.append(1)
    if i>=len(partial_test_list1) and i<len(partial_test_list1)+len(partial_test_list2):
        acc_checker.append(0)
    if i>=len(partial_test_list1)+len(partial_test_list2) and i<len(partial_test_list1)+len(partial_test_list2)+len(partial_test_list4):
        acc_checker.append(0)
    if i>=len(partial_test_list1)+len(partial_test_list2)+len(partial_test_list4) and i<len(partial_test_list1)+len(partial_test_list2)+len(partial_test_list4)+len(partial_test_list5):
        acc_checker.append(1)
acc=0
for i, label in enumerate(cluster_labels):
    if acc_checker[i]==label:
        acc+=1
print(acc/len(full_partial_test))        

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embedding)  # now shape (n_graphs, 2)

# 4) Plot
plt.figure(figsize=(10, 6))
for i, (x, y) in enumerate(embeddings_2d):
    color = 'red' if cluster_labels[i] == 0 else 'blue'
    correct = (cluster_labels[i] == acc_checker[i])
    marker = 'o' if correct else 'x'
    plt.scatter(x, y, c=color, marker=marker, s=100,
                label=('Correct' if correct else 'Incorrect') if i == 0 else "",
                alpha=0.7)
    plt.text(x + 0.5, y + 0.5, str(i), fontsize=8, alpha=0.6)

plt.title("t-SNE Visualization of Graph Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# ==== Corrected KMeans Clustering with Full 2D Embeddings ====
print("\n==== Re-running clustering with full 2D embeddings ====")

# Get full 2D embeddings for datas
embedding_datas, _ = get_embeddings(model, datas)

# Convert to float64 for KMeans compatibility
embeddings_datas = embedding_datas.astype(np.float64)

# Fit KMeans on 2D embeddings
k = 2
kmeans = KMeans(n_clusters=k, random_state=0)
cluster_labels = kmeans.fit_predict(embeddings_datas)

# Calculate accuracy
acc_checker = []
for i in range(len(datas)):
    if i < len(l2): 
        acc_checker.append(0)
    elif i < len(l2)+len(l3):
        acc_checker.append(1)
    else: 
        acc_checker.append(0)
    
acc = sum(1 for i, label in enumerate(cluster_labels) if acc_checker[i] == label)
print(f"Accuracy with 2D embeddings: {acc/len(datas):.2f}")

# ==== Updated Prediction Functions ====
def predict_distances(embs):
    """Returns negative distances to cluster centroids"""
    # Ensure input is float64 for KMeans compatibility
    return -kmeans.transform(embs.astype(np.float64))

def predict_fn_for_lime_nodes(perturbed_node_masks_batch, 
                             graph_being_explained_x, 
                             graph_being_explained_edge_index, 
                             model_to_use, 
                             kmeans_predictor_func):
    """Generates KMeans distances for perturbed graphs"""
    batch_predictions = []
    original_x = graph_being_explained_x.clone().detach()
    original_edge_index = graph_being_explained_edge_index.clone().detach()
    
    model_to_use.eval()
    
    for i in range(perturbed_node_masks_batch.shape[0]):
        mask = torch.tensor(perturbed_node_masks_batch[i, :], 
                           dtype=torch.float).unsqueeze(1)
        
        # Apply perturbation with noise for stability
        perturbed_x = original_x * mask + torch.randn_like(original_x) * 0.01
        
        with torch.no_grad():
            emb_perturbed, _ = model_to_use(perturbed_x, original_edge_index)
        
        # Keep as 2D: [1, 2] shape and convert to float64
        emb_perturbed_np = emb_perturbed.squeeze().cpu().numpy().astype(np.float64)
        emb_2d = emb_perturbed_np.reshape(1, -1)  # Ensure 2D shape
        
        distances = kmeans_predictor_func(emb_2d)
        batch_predictions.append(distances[0])
        
    return np.array(batch_predictions)

# ==== LIME Setup and Explanation ====
print("\n==== Setting up LIME explainer ====")

# Select a graph to explain
graph_idx_to_explain = 0
graph_to_explain_pyg = training_list[graph_idx_to_explain]
num_nodes = len(ref_node_names)

# Create baseline dataset (binary node presence)
lime_training_data = np.random.randint(0, 2, size=(500, num_nodes))

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=lime_training_data,
    feature_names=ref_node_names,
    class_names=[f"Cluster_{i}" for i in range(k)],
    mode='regression',
    discretize_continuous=False
)

# Create specialized prediction function
predict_fn = partial(predict_fn_for_lime_nodes,
                     graph_being_explained_x=graph_to_explain_pyg.x,
                     graph_being_explained_edge_index=graph_to_explain_pyg.edge_index,
                     model_to_use=model,
                     kmeans_predictor_func=predict_distances)

# Generate explanation
print(f"\n==== Explaining graph {graph_idx_to_explain} ====")
exp = explainer.explain_instance(
    data_row=np.ones(num_nodes),  # All nodes present
    predict_fn=predict_fn,
    num_features=num_nodes,
    num_samples=1000,
    top_labels=k
)

# ==== Display Results ====
print("\n==== Explanation Results ====")

# Get actual cluster prediction
with torch.no_grad():
    emb_actual, _ = model(graph_to_explain_pyg.x, graph_to_explain_pyg.edge_index)
    
# Convert to float64 and ensure 2D shape
emb_2d_actual = emb_actual.squeeze().cpu().numpy().astype(np.float64).reshape(1, -1)
actual_cluster = kmeans.predict(emb_2d_actual)[0]
print(f"Graph belongs to Cluster {actual_cluster}")

# Show explanations per cluster
for cluster_id in range(k):
    print(f"\nFeature contributions to distance from Cluster {cluster_id} centroid:")
    print("(Positive = moves graph closer to this cluster)")
    print("(Negative = moves graph away from this cluster)")
    
    # Get explanations sorted by absolute value
    expl_list = exp.as_list(label=cluster_id)
    expl_list_sorted = sorted(expl_list, key=lambda x: abs(x[1]), reverse=True)
    
    for feature, weight in expl_list_sorted[:15]:  # Top 15 features
        print(f"{feature:>20}: {weight:>8.4f}")

# ==== Visualize Important Nodes ====
print("\nVisualizing important nodes...")

def visualize_important_nodes(exp, cluster_id, top_n=5):
    """Highlight top influential nodes for a cluster"""
    expl_list = exp.as_list(label=cluster_id)
    top_nodes = sorted(expl_list, key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    print(f"\nTop {top_n} nodes influencing distance to Cluster {cluster_id}:")
    for node, weight in top_nodes:
        influence = "↑ CLOSER" if weight > 0 else "↓ AWAY"
        print(f"{node:>20}: {weight:>8.4f} ({influence})")

# Visualize for actual cluster
visualize_important_nodes(exp, actual_cluster)

# Visualize for opposite cluster
opposite_cluster = 1 - actual_cluster
visualize_important_nodes(exp, opposite_cluster)
