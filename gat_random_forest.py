import os
import statistics
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, GINEConv, GATConv
from torch_cluster import knn_graph, graclus
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from captum.attr import IntegratedGradients
    
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
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        return global_mean_pool(x, batch)

# Initialize model and optimizer
model = GNN(input_dim=ref_graph_data.x.size(1), hidden_dim=16, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
ref_embedding = model(ref_graph_data.x, ref_graph_data.edge_index)
ref_embedding = ref_embedding.detach()
for epoch in range(1000):
    total_loss = 0
    optimizer.zero_grad()
    for data in training_list:
        adj_recon = model(data.x, data.edge_index)
        loss = F.mse_loss(adj_recon, ref_embedding)
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(training_list)}")

def get_embeddings(model, graph_list):
    embeddings = []
    for graph in graph_list:
        with torch.no_grad():
            emb = model(graph.x, graph.edge_index)
        embeddings.append(emb.squeeze().detach().numpy())
    return np.array(embeddings)

embedding = get_embeddings(model, datas)
embeddings = embedding.mean(axis=1)
embeddings = embeddings.reshape(-1, 1)
embeddings = embeddings.astype(np.float64)

# Create labels for training
acc_checker=[]
for i in range(len(datas)):
    if i<len(l2):
        acc_checker.append(0)
    if i>=len(l2) and i<len(l2)+len(l3):
        acc_checker.append(1)
    if i>=len(l2)+len(l3) and i<len(l2)+len(l3)+len(l4):
        acc_checker.append(0)

# Now run Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(embeddings, acc_checker)
cluster_labels = rf.predict(embeddings)
print(cluster_labels)

from sklearn.metrics import accuracy_score, precision_score, recall_score

acc = accuracy_score(acc_checker, cluster_labels)
prec = precision_score(acc_checker, cluster_labels, zero_division=0)
rec = recall_score(acc_checker, cluster_labels, zero_division=0)

res_df = pd.DataFrame({"File": ["gat_random_forest.py"], "Technique": ["GAT + Random Forest"], "Accuracy": [acc], "Precision": [prec], "Recall": [rec]})
import os
try:
    if os.path.exists("metrics_results.xlsx"):
        with pd.ExcelWriter("metrics_results.xlsx", mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            res_df.to_excel(writer, startrow=writer.sheets["Sheet1"].max_row, index=False, header=False)
    else:
        res_df.to_excel("metrics_results.xlsx", index=False)
except Exception as e:
    print(e)
print("Metrics saved.")


