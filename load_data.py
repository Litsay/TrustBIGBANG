import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import TrustBIGBANG_model
import networkx as nx
import os

def label_to_num(label):
    if label in ['E13', 'TFP'] :
        num = 1
    elif label in ['INT', 'FSF', 'TWT']:
        num = 2
    else:
        num = 3
    return num

def load_node_csv(paths, index_col, node_type, encoders=None, **kwargs):
    df = pd.read_csv(paths, index_col=index_col)
    print('loading:', paths)
    
    item_id = df.index
    mapped_item_id = {index: i for i, index in enumerate(item_id.unique())}
    
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapped_item_id

class SequenceEncoder(object):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        df = df.fillna(' ')
        x = self.model.encode(df.values, show_progress_bar=True, convert_to_tensor=True, device=self.device)
        return x.cpu()    

class NumberEncoder(object):
    def __call__(self, df):
        df = df.fillna(0)
        x = torch.zeros(len(df), 1)
        i = 0
        for item in df:
            x[i][0] = item
            i = i+1
        return x
       

class LabelEncoder(object):

    def __call__(self, df):
        y = torch.zeros(len(df), 1)
        i = 0
        for item in df:
            y[i][0] = label_to_num(str(item))
            i = i+1
        return y




def load_edge_csv(path, edge_type, mapped_dst_id_type, src_index_col, src_id, dst_index_col, dst_id, encoders=None, **kwargs):
    df = pd.read_csv(path, dtype={'id':np.int64}, sep=' ')
    print('loading:', path)
    #print(df)
 
    #df.insert(2, 'type', edge_type)
    
    print(df.size)
    
    #df[dst_index_col] = df[dst_index_col].astype('int64')
    src = df[src_index_col]
    dst = df[dst_index_col]
    #dst = [mapped_dst_id_type.get(index) for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    
    edge_attr = None
    if encoders is not None:
        #edge_attr = df['trust_value']
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)
    
    return edge_index.long(), edge_attr



class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype
    
    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtpye)
    
class NumberEncoder(object):
    def __call__(self, df):
        df = df.fillna(0)
        x = torch.zeros(len(df), 1)
        i = 0
        print(df)
        for item in df:
            if item == 0.6:
                x[i][0] = 0
            elif item == 0.8:
                x[i][0] = 1
            else:
                x[i][0] = 2
            i = i+1
        #rint(x)
        return x