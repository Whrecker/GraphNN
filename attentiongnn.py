import pandas as pd
import os
import statistics
import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

#add explainable ai
class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_scores = torch.sigmoid(self.attn(x))  # Compute attention scores
        attn_x = x * attn_scores  # Apply attention
        return attn_x
'''
class WeightedMessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(WeightedMessagePassing, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, node_importance):
        # Propagate messages while passing the node importance.
        return self.propagate(edge_index, x=x, node_importance=node_importance)
    
    def message(self, x_j, node_importance):
        # Multiply each neighbor's message by its corresponding importance weight.
        return self.lin(x_j) * node_importance.view(-1, 1)
'''

# 🔹 Updated Graph Encoder with Attention
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim)  # Use GAT for better learning
        self.conv2 = GATConv(hidden_dim, output_dim)
        self.attn = AttentionModule(output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        #x = self.attn(x)  # Apply attention module
        x = global_mean_pool(x, batch=None)  
        return x

# 🔹 Main Model
class G2GSimilarityNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(G2GSimilarityNet, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim * 2, 1)

    def forward(self, g1, g2):
        x1 = self.encoder(g1.x, g1.edge_index, g1.edge_attr)
        x2 = self.encoder(g2.x, g2.edge_index, g2.edge_attr)
        x_concat = torch.cat([x1, x2], dim=1)
        similarity = self.fc(x_concat)
        return similarity


def fit_one_hot_encoder(reference_graph):
    """Fits separate OneHotEncoders for node names and locations."""
    node_names = [node[0] for node in reference_graph['nodes']]
    locations = [node[-1] for node in reference_graph['nodes']]  # Assuming location is last

    node_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    location_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    node_encoder.fit(np.array(node_names).reshape(-1, 1))
    location_encoder.fit(np.array(locations).reshape(-1, 1))

    return node_encoder, location_encoder

def create_graph_data(graph, node_encoder, location_encoder, ref_node_names):
    nodes = graph['nodes']
    edges = graph['edges']

    node_names = [node[0] for node in nodes]
    node_attributes = {node[0]: node[1:] for node in nodes}  

    # Get expected feature sizes
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
            encoded_name = np.zeros(name_dim)  # Zero vector for missing nodes
            attributes = [0] * (len(nodes[0]) - 2)  
            encoded_location = np.zeros(location_dim)  

        full_features = np.hstack([encoded_name, attributes, encoded_location])

        # Debug: Check if all vectors have the same length
        if len(full_features) != (name_dim + len(attributes) + location_dim):
            raise ValueError(f"Inconsistent feature length for node {name}: Expected {name_dim + len(attributes) + location_dim}, Got {len(full_features)}")

        node_features.append(full_features)

    node_features = torch.tensor(node_features, dtype=torch.float)

    # Edge processing (unchanged)
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


# Load data
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

# Initialize model, optimizer, and loss function
model = G2GSimilarityNet(input_dim=ref_graph_data.x.size(1), hidden_dim=128, output_dim=16)
model2 = G2GSimilarityNet(input_dim=ref_graph_data.x.size(1), hidden_dim=128, output_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()
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
node_importance = torch.ones(ref_graph_data.x.size(0))
node_importance[0] = 2.0


for epoch in range(10001):
    model.train()
    optimizer.zero_grad()

    # Compute positive similarity
    pos_similarity = [model(ref_graph_data,i) for i in training_list]

    # Compute negative similarities
    neg_similarities = [model(ref_graph_data, i) for i in other_graph_data]

    # Aggregate all predictions
    predictions = torch.cat(pos_similarity + neg_similarities, dim=0).squeeze(1)
    
    labels = torch.tensor([1]*len(pos_similarity)+[0]*len(neg_similarities),dtype=float)
    loss = criterion(predictions, labels)
    loss = (loss * node_importance.view(-1, 1)).mean()
    loss.backward()
    optimizer.step()
    # Logging every 100 epochs
    if epoch % 100 == 0:
        print(predictions)
        print(labels)
        with torch.no_grad():
            test_score_1 = torch.sigmoid(model(ref_graph_data, test1)).item()
            test_score_2 = torch.sigmoid(model(ref_graph_data, test2)).item()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test1={round(test_score_1, 2)}, Test2={round(test_score_2, 2)}")
    if loss.item()<=0.1:
        break
acc_counter=0
counter=0
model.eval()
with torch.no_grad():
    for i in range(len(datas)):
        if i>len(l2)+len(l3):
            counter+=1
            score=round(torch.sigmoid(model(ref_graph_data,datas[i])).item(),2)
            if score>.5:
                acc_counter+=1
            print("should be above .5",score)
        if i>=len(l2) and i<len(l2)+len(l3):
            counter+=1
            score=round(torch.sigmoid(model(ref_graph_data,datas[i])).item(),2)
            if score<.5:
                acc_counter+=1
            print("should be below .5",score)
print(acc_counter/counter)

print(round(torch.sigmoid(model(ref_graph_data,test1)).item(),2))
print(round(torch.sigmoid(model(ref_graph_data,test2)).item(),2))


print("partial testing")
acc_counter=0
for i in range(len(partial_test_list1)):
    print(i)
    '''    
    score=round(torch.sigmoid(model(ref_graph_data,partial_test_list1[i])).item(),2)
    if score>.5:
        print("should be more than .5",round(torch.sigmoid(model(ref_graph_data,partial_test_list1[i])).item(),2))
        acc_counter+=1
    score=round(torch.sigmoid(model(ref_graph_data,partial_test_list2[i])).item(),2)
    if score<.5:
        print("should be less than .5",round(torch.sigmoid(model(ref_graph_data,partial_test_list2[i])).item(),2))
        acc_counter+=1
    ''' 
    score=round(torch.sigmoid(model(ref_graph_data_list[i],partial_test_list5[i])).item(),2)
    print("should be more than .5",round(torch.sigmoid(model(ref_graph_data,partial_test_list5[i])).item(),2))
    if score>.5:
        acc_counter+=1

    score=round(torch.sigmoid(model(ref_graph_data_list[i],partial_test_list4[i])).item(),2)
    print("should be less than .5",round(torch.sigmoid(model(ref_graph_data,partial_test_list4[i])).item(),2))
    if score<.5:
        acc_counter+=1
print(acc_counter/(len(partial_test_list1)*2))


for epoch in range(10001):
    model2.train()
    optimizer2.zero_grad()

    # Compute positive similarity
    pos_similarity = [model2(partial_ref_graph_data,i) for i in partial_test_list3]

    # Compute negative similarities
    neg_similarities = [model2(partial_ref_graph_data, i) for i in partial_test_list4]

    # Aggregate all predictions
    predictions = torch.cat(pos_similarity + neg_similarities, dim=0).squeeze(1)
    
    labels = torch.tensor([1]*len(pos_similarity)+[0]*len(neg_similarities),dtype=float)
    loss = criterion(predictions, labels)
    loss = (loss * node_importance.view(-1, 1)).mean()
    loss.backward()
    optimizer2.step()
    # Logging every 100 epochs
    if epoch % 100 == 0:
        print(predictions)
        print(labels)
        with torch.no_grad():
            test_score_1 = torch.sigmoid(model2(ref_graph_data, test1)).item()
            test_score_2 = torch.sigmoid(model2(ref_graph_data, test2)).item()
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Test1={round(test_score_1, 2)}, Test2={round(test_score_2, 2)}")
    if loss.item()<=0.1:
        break

print("partial testing")
acc_counter=0
for i in range(len(partial_test_list1)):
    print(i)
    '''    
    score=round(torch.sigmoid(model(ref_graph_data,partial_test_list1[i])).item(),2)
    if score>.5:
        print("should be more than .5",round(torch.sigmoid(model(ref_graph_data,partial_test_list1[i])).item(),2))
        acc_counter+=1
    score=round(torch.sigmoid(model(ref_graph_data,partial_test_list2[i])).item(),2)
    if score<.5:
        print("should be less than .5",round(torch.sigmoid(model(ref_graph_data,partial_test_list2[i])).item(),2))
        acc_counter+=1
    ''' 
    score=round(torch.sigmoid(model2(ref_graph_data_list[i],partial_test_list5[i])).item(),2)
    print("should be more than .5",round(torch.sigmoid(model2(ref_graph_data,partial_test_list5[i])).item(),2))
    if score>.5:
        acc_counter+=1

    score=round(torch.sigmoid(model2(ref_graph_data_list[i],partial_test_list4[i])).item(),2)
    print("should be less than .5",round(torch.sigmoid(model2(ref_graph_data,partial_test_list4[i])).item(),2))
    if score<.5:
        acc_counter+=1
print(acc_counter/(len(partial_test_list1)*2))


def model_wrapper(x_features):
    modified_test = Data(x=x_features, edge_index=datas[0].edge_index, edge_attr=datas[0].edge_attr)
    return model(ref_graph_data, modified_test)

# Use a baseline of zeros with the same shape as test1.x.
baseline = torch.zeros_like(datas[0].x)
input_features = datas[0].x.clone()

ig = IntegratedGradients(model_wrapper)
attributions, delta = ig.attribute(input_features, baseline, target=0, return_convergence_delta=True)
print('IG Attributions:', attributions)
print('Convergence Delta:', delta)
avg_attr = attributions.abs().mean(dim=0)
print(avg_attr)


