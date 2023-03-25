import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import networkx as nx
import os
import time
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, HypergraphConv
from torch.nn import Linear
from torch import nn
torch.cuda.current_device()


import load_data
import graph_construction
import warnings
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index, data.hyperedge_index)
    edge_label_index = data.edge_index
    edge_label = data.edge_label
    out_pred, out_prob, out = model.decode(z, edge_label_index)

    #print('out_pred:', out_pred)
    #print('out_prob:', out_prob)
    #print('edge_label', edge_label)
    mask = data.train_mask
    #print(out_prob[mask])
    #loss = F.cross_entropy(out[mask], data.edge_label[mask])
    loss = F.cross_entropy(out[extend_train_mask], data.edge_label[extend_train_mask])
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data.x, data.edge_index, data.hyperedge_index)
    out_pred, out_prob, out = model.decode(z, data.edge_index)
    #print(out_pred)
    #print(data.edge_label)
    accs = []
    prc_scores = []
    rca_scores = []
    f1_scores = []

    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data[split]
        acc = (out_pred[mask] == data.edge_label[mask].long()).sum() / mask.sum()
        accs.append(float(acc))
        
        prc = precision_score(data.edge_label[mask].long().cpu().numpy(), out_pred[mask].cpu().numpy(), average='macro')
        prc_scores.append(float(prc))
                              
        rca = recall_score(data.edge_label[mask].long().cpu().numpy(), out_pred[mask].cpu().numpy(), average='macro')
        rca_scores.append(float(rca))                      
                              
        f1 = f1_score(data.edge_label[mask].long().cpu().numpy(), out_pred[mask].cpu().numpy(), average='macro')
        f1_scores.append((f1))
        
    return list(accs), list(prc_scores), list(rca_scores), list(f1_scores)





if __name__ == '__main__':

    # data loading

    trust_edge_path = r'./advogato/advogato.txt'
    user_name_path = r'./advogato/ent.advogato.user.name'
    user_data_path = r'./advogato/advogato-user.csv'
    raw_data = pd.read_csv(trust_edge_path, sep=' ')
    print(raw_data)

    from_node = raw_data.iloc[:,0]
    to_node = raw_data.iloc[:,1]
    trust_value = raw_data.iloc[:,2]


    user_list = []
    for node in from_node:
        if node not in user_list:
            user_list.append(node)
    print(len(user_list))
    for node in to_node:
        if node not in user_list:
            user_list.append(node)
    print(len(user_list))

    for node in range(1,6541):
        if node not in user_list:
            print("isolated node: No.", node)    
    user_list.sort()


    print("trust_value==0.6: ", (trust_value==0.6).sum())
    print("trust_value==0.8: ", (trust_value==0.8).sum())
    print("trust_value==1: ", (trust_value==1).sum())


    user_name = pd.read_csv(user_name_path)

    user_id = np.empty(6541, dtype = int)
    for i in range(6541):
        user_id[i] = i+1

    user_id = pd.DataFrame(user_id, columns=['user_id'])
    user_data = pd.concat([user_id, user_name],axis=1)
    print(user_data)
    user_data.to_csv(user_data_path, index=False)



    # loading user nodes
    df = pd.read_csv(user_data_path, index_col='user_id')
    user_x, user_id = load_data.load_node_csv(
        user_data_path, index_col='user_id', node_type='user', encoders={
            'user_name': load_data.SequenceEncoder(),
        })
    print(user_x.size())
    print(user_x)


    from torch_geometric.data import Data
    data = Data()
    data.x = user_x
    print(data)


    # loading trust relations
    edge_index, edge_label = load_data.load_edge_csv(
        trust_edge_path,
        edge_type=1,
        mapped_dst_id_type=user_id,
        src_index_col='source_id',
        src_id=user_id,
        dst_index_col='target_id',
        dst_id=user_id,
        encoders={'trust_value': load_data.NumberEncoder()},
    )
    print(edge_index)
    #print(edge_label)
    #print(edge_index.shape)

    data.edge_index = edge_index
    data.edge_label = edge_label.long().ravel()
    print(data)
    print(data.edge_label)

    # check
    data.edge_index.max() < data.x.size(0)
    #print(data.edge_index.max())
    #print(data.x.size(0))

    # get data musk
    train_mask, val_mask, test_mask = graph_construction.get_mask(data.edge_label.size()[0], 0.02, 0.02)
    extend_train_mask = train_mask

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(data.train_mask)
    print(data.val_mask)
    print(data.test_mask)
    print(data.edge_label)
    print(extend_train_mask)
    print(data)


    G = graph_construction.load_graph(trust_edge_path)
    obj = graph_construction.Graph()
    start_time = time.time()
    algorithm = graph_construction.Louvain(G)
    communities = algorithm.execute()
    end_time = time.time()
    communities = sorted(communities, key=lambda b: -len(b))
    count = 0
    hyperedge = []
    for communitie in communities:
        count += 1
        if len(communitie) > 2:
            hyperedge.append(communitie)
            print("Hyper-relation:", count, " ", communitie, "user included：", len(communitie))

    #print(cal_Q(communities, G1))
    print(f'Hyper-relation exacting module running time{end_time - start_time}')
    print('Hyperedge exacted:', hyperedge)



    regular_edge = []
    for counter in range(0, len(from_node)):
        edge = [from_node[counter], to_node[counter]]
        regular_edge.append(edge)
        
    #regular_edge.extend(hyperedge)
    hyperedge.extend(regular_edge)
    #print(regular_edge)

    def edge_transform(edge_list):
        edge_num = []
        edge_index = []
        num = 0
        for edge in edge_list:
            edge_index.extend(edge)
            for i in range(0, len(edge)):
                edge_num.append(int(num))
            num = num+1
        hyperedge_index = [edge_index, edge_num]
        return hyperedge_index

    hyperedge_index = edge_transform(hyperedge)
    data.hyperedge_index = torch.tensor(hyperedge_index)
    #print(hyperedge_index)

    
    weight = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)


    model = graph_construction.Net(-1, 128, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0002, weight_decay=0.005)

    Best_F1 = 0
    for epoch in range(1, 3001):
        loss = train()
        accs, prc_scores, rca_scores, f1_scores = test()
        if Best_F1 < f1_scores[2]:
            Best_F1 = f1_scores[2]
        print(f'Epoch: {epoch:03d}, Loss: {loss:.5f}, Train: {accs[0]:.5f}, Val: {accs[1]:.5f},'
            f'Test: {accs[2]:.5f}, Precision: {prc_scores[2]:.5f}, Recall: {rca_scores[2]:.5f}, F1-score: {f1_scores[2]:.5f}, Best F1-socre: {Best_F1:.5f}')
