import math
import random
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import collections
import random
import time
import networkx as nx
import pandas as pd
import TrustBIGBANG_model
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HypergraphConv
from torch_geometric.nn import MLP

import math
import random
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_mask(all_num, val_rate, test_rate):
    val_num = math.floor(all_num*val_rate)
    test_num = math.floor(all_num*test_rate)
    train_num = all_num - val_num - test_num
    all_mask = np.random.randint(1, size=train_num).tolist() + np.random.randint(1, 2, size=val_num).tolist() + np.random.randint(2, 3, size=test_num).tolist()
    #all_musk = random.shuffle()
    shuffled_mask = random.shuffle(all_mask)
    
    train_mask = torch.tensor([True if item == 0 else False for item in all_mask], dtype=torch.bool)
    val_mask = torch.tensor([True if item == 1 else False for item in all_mask], dtype=torch.bool)
    test_mask = torch.tensor([True if item == 2 else False for item in all_mask], dtype=torch.bool)
    
    return train_mask, val_mask, test_mask






def load_graph(path):
    trust_edge_path = r'./advogato/advogato.txt'
    user_name_path = r'./advogato/ent.advogato.user.name'
    user_data_path = r'./advogato/advogato-user.csv'
    raw_data = pd.read_csv(trust_edge_path, sep=' ')
    from_node = raw_data.iloc[:,0]
    to_node = raw_data.iloc[:,1]
    trust_value = raw_data.iloc[:,2]

    G = collections.defaultdict(dict)
    for counter in range(0, len(from_node)):
        v_i = int(from_node[counter])
        v_j = int(to_node[counter])
        w = 1.0
        G[v_i][v_j] = w
        G[v_j][v_i] = w
    return G



class Vertex:
    def __init__(self, vid, cid, nodes, k_in=0):
        self._vid = vid
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in


class Louvain:
    def __init__(self, G):
        self._G = G
        self._m = 0 
        self._cid_vertices = {}
        self._vid_vertex = {}
        for vid in self._G.keys():
            self._cid_vertices[vid] = {vid}
            self._vid_vertex[vid] = Vertex(vid, vid, {vid})
            self._m += sum([1 for neighbor in self._G[vid].keys()
                           if neighbor > vid])


    def first_stage(self):
        mod_inc = False  
        visit_sequence = self._G.keys()
        random.shuffle(list(visit_sequence))
        while True:
            can_stop = True 
            for v_vid in visit_sequence:
                v_cid = self._vid_vertex[v_vid]._cid
                k_v = sum(self._G[v_vid].values()) + \
                    self._vid_vertex[v_vid]._kin
                cid_Q = {}
                for w_vid in self._G[v_vid].keys():
                    w_cid = self._vid_vertex[w_vid]._cid
                    if w_cid in cid_Q:
                        continue
                    else:
                        tot = sum(
                            [sum(self._G[k].values()) + self._vid_vertex[k]._kin for k in self._cid_vertices[w_cid]])
                        if w_cid == v_cid:
                            tot -= k_v
                        k_v_in = sum(
                            [v for k, v in self._G[v_vid].items() if k in self._cid_vertices[w_cid]])
                        delta_Q = k_v_in - k_v * tot / self._m
                        cid_Q[w_cid] = delta_Q

                
                cid, max_delta_Q = sorted(
                    cid_Q.items(), key=lambda item: item[1], reverse=True)[0]
                if max_delta_Q > 0.0 and cid != v_cid:
                    self._vid_vertex[v_vid]._cid = cid
                    self._cid_vertices[cid].add(v_vid)
                    self._cid_vertices[v_cid].remove(v_vid)
                    can_stop = False
                    mod_inc = True
            if can_stop:
                break
        return mod_inc


    def second_stage(self):
        cid_vertices = {}
        vid_vertex = {}
        for cid, vertices in self._cid_vertices.items():
            if len(vertices) == 0:
                continue
            new_vertex = Vertex(cid, cid, set())
            for vid in vertices:
                new_vertex._nodes.update(self._vid_vertex[vid]._nodes)
                new_vertex._kin += self._vid_vertex[vid]._kin
                for k, v in self._G[vid].items():
                    if k in vertices:
                        new_vertex._kin += v / 2.0
            cid_vertices[cid] = {cid}
            vid_vertex[cid] = new_vertex

        G = collections.defaultdict(dict)
        for cid1, vertices1 in self._cid_vertices.items():
            if len(vertices1) == 0:
                continue
            for cid2, vertices2 in self._cid_vertices.items():
                if cid2 <= cid1 or len(vertices2) == 0:
                    continue
                edge_weight = 0.0
                for vid in vertices1:
                    for k, v in self._G[vid].items():
                        if k in vertices2:
                            edge_weight += v
                if edge_weight != 0:
                    G[cid1][cid2] = edge_weight
                    G[cid2][cid1] = edge_weight
        self._cid_vertices = cid_vertices
        self._vid_vertex = vid_vertex
        self._G = G

    def get_communities(self):
        communities = []
        for vertices in self._cid_vertices.values():
            if len(vertices) != 0:
                c = set()
                for vid in vertices:
                    c.update(self._vid_vertex[vid]._nodes)
                communities.append(list(c))
        return communities

    def execute(self):
        iter_time = 1
        while True:
            iter_time += 1
            mod_inc = self.first_stage()
            if mod_inc:
                self.second_stage()
            else:
                break
        return self.get_communities()


def cal_Q(partition, G):  # 计算Q
    m = len(G.edges(None, False))
    # print(G.edges(None,False))
    a = []
    e = []
    for community in partition: 
        t = 0.0
        for node in community: 
            t += len([x for x in G.neighbors(node)])
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q

class Graph:
    graph = nx.DiGraph()

    def __init__(self):
        self.graph = nx.DiGraph()

    def createGraph(self, filename):
        file = open(filename, 'r')

        for line in file.readlines():
            nodes = line.split()
            edge = (int(nodes[0]), int(nodes[1]))
            self.graph.add_edge(*edge)

        return self.graph


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, out_channels)
        self.mlp = MLP(in_channels=128,out_channels=3,hidden_channels=64,num_layers=4,dropout=0.5)

    def encode(self, x, edge_index, hyperedge_index):
        x = self.conv1(x, hyperedge_index).relu_()
        x = self.conv2(x, edge_index).relu_()
        return self.conv3(x, edge_index).relu_()
        

    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        edge_attr = torch.cat([z[src], z[dst]], dim=-1)
        #print(edge_attr.size())
        out = self.mlp(edge_attr).to(device)
        out_pred = torch.argmax(self.mlp(edge_attr), -1).to(device)
        prob_list = []
        for index in range(0, len(out_pred)):
            prob_list.append(out[index][out_pred[index]])
        out_prob = torch.tensor(prob_list)
        #print('out_pred:', out_pred)
        #print('out_prob:', out_prob)
        return out_pred, out_prob, out


