import os
import csv
import pickle
import numpy as np
import json
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from typing import Callable, Optional, Union, List
from sklearn.cluster import AgglomerativeClustering
from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import k_hop_subgraph, from_networkx
from pyhealth.tokenizer import Tokenizer
from pyhealth.models.utils import batch_to_multihot
from pyhealth.datasets import split_by_patient


def load_dataset(dataset, task, sample_dataset, batch_size):
    ent2id, rel2id, ent_emb, rel_emb = load_embeddings()
    ccscm_id2clus, ccsproc_id2clus, atc3_id2clus, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel = clustering(ent_emb, rel_emb, threshold=0.15)
    G_tg = process_graph(dataset, sample_dataset, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel)
    if os.path.exists(f'resources/graphcare/{dataset}_dataset.pkl'):
        with open(f'resources/graphcare/{dataset}_dataset.pkl', 'rb') as f:
            loaded_sample_dataset = pickle.load(f)
        for i in range(len(sample_dataset)):
            for name in ["label", "text", "perplexity", "embedding"]:
                if name in sample_dataset[i]:
                    loaded_sample_dataset[i][name] = sample_dataset[i][name]
        sample_dataset = loaded_sample_dataset
    else:
        sample_dataset = process_sample_dataset(sample_dataset, G_tg, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel)
        sample_dataset = label_ehr_nodes(sample_dataset, len(map_cluster), ccscm_id2clus, ccsproc_id2clus, atc3_id2clus)
        with open(f'resources/graphcare/{dataset}_dataset.pkl', 'wb') as f:
            pickle.dump(sample_dataset, f)
        
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1], seed=2
    )

    node_emb = G_tg.x
    rel_emb = get_rel_emb(map_cluster_rel)

    # get dataloader
    train_dataset = Dataset(G_tg, train_dataset)
    val_dataset = Dataset(G_tg, val_dataset)
    test_dataset = Dataset(G_tg, test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, node_emb, rel_emb, sample_dataset


def load_embeddings():
    path = "resources/graphcare/cond_proc_drug/CCSCM_CCSPROC_ATC3"
    with open(f'{path}/ent2id.json', 'r') as file:
        ent2id = json.load(file)
    with open(f'{path}/rel2id.json', 'r') as file:
        rel2id = json.load(file)
    with open(f'{path}/entity_embedding.pkl', 'rb') as file:
        ent_emb = pickle.load(file)
    with open(f'{path}/relation_embedding.pkl', 'rb') as file:
        rel_emb = pickle.load(file)
    return ent2id, rel2id, ent_emb, rel_emb


def clustering(ent_emb, rel_emb, threshold=0.15):
    path = "resources/graphcare/ccscm_ccsproc_atc3"

    if os.path.exists(f'{path}/clusters_th015.json'):
        with open(f'{path}/clusters_th015.json', 'r', encoding='utf-8') as f:
            map_cluster = json.load(f)
        with open(f'{path}/clusters_inv_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv = json.load(f)
        with open(f'{path}/clusters_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_rel = json.load(f)
        with open(f'{path}/clusters_inv_rel_th015.json', 'r', encoding='utf-8') as f:
            map_cluster_inv_rel = json.load(f)
        with open(f'{path}/ccscm_id2clus.json', 'r') as f:
            ccscm_id2clus = json.load(f)
        with open(f'{path}/ccsproc_id2clus.json', 'r') as f:
            ccsproc_id2clus = json.load(f)
        with open(f'{path}/atc3_id2clus.json', 'r') as f:
            atc3_id2clus = json.load(f)

    else:
        cluster_alg = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average', affinity='cosine')
        cluster_labels = cluster_alg.fit_predict(ent_emb)
        cluster_labels_rel = cluster_alg.fit_predict(rel_emb)

        def nested_dict():
            return defaultdict(list)

        map_cluster = defaultdict(nested_dict)

        for unique_l in np.unique(cluster_labels):
            for cur in range(len(cluster_labels)):
                if cluster_labels[cur] == unique_l:
                    map_cluster[str(unique_l)]['nodes'].append(cur)

        for unique_l in map_cluster.keys():
            nodes = map_cluster[unique_l]['nodes']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster[unique_l]['embedding'].append(embedding_mean.tolist())

        map_cluster_inv = {}
        for cluster_label, item in map_cluster.items():
            for node in item['nodes']:
                map_cluster_inv[str(node)] = cluster_label

        map_cluster_rel = defaultdict(nested_dict)

        for unique_l in np.unique(cluster_labels_rel):
            for cur in range(len(cluster_labels_rel)):
                if cluster_labels_rel[cur] == unique_l:
                    map_cluster_rel[str(unique_l)]['relations'].append(cur)

        for unique_l in map_cluster_rel.keys():
            nodes = map_cluster_rel[unique_l]['relations']
            nodes = np.array(nodes)
            embedding_mean = np.mean(ent_emb[nodes], axis=0)
            map_cluster_rel[unique_l]['embedding'].append(embedding_mean.tolist())

        map_cluster_inv_rel = {}
        for cluster_label, item in map_cluster_rel.items():
            for node in item['relations']:
                map_cluster_inv_rel[str(node)] = cluster_label

        with open(f'{path}/clusters_th015.json', 'w', encoding='utf-8') as f:
            json.dump(map_cluster, f, indent=6)
        with open(f'{path}/clusters_inv_th015.json', 'w', encoding='utf-8') as f:
            json.dump(map_cluster_inv, f, indent=6)
        with open(f'{path}/clusters_rel_th015.json', 'w', encoding='utf-8') as f:
            json.dump(map_cluster_rel, f, indent=6)
        with open(f'{path}/clusters_inv_rel_th015.json', 'w', encoding='utf-8') as f:
            json.dump(map_cluster_inv_rel, f, indent=6)
            
            
        ccscm_id2name = {}
        with open('resources/CCSCM.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split(',')
                ccscm_id2name[line[0]] = line[1].lower()

        ccsproc_id2name = {}
        with open('resources/CCSPROC.csv', 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip().split(',')
                ccsproc_id2name[line[0]] = line[1].lower()

        atc3_id2name = {}
        with open("resources/ATC.csv", newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['level'] == '3.0':
                    atc3_id2name[row['code']] = row['name'].lower()
        ccscm_id2emb = {}
        ccsproc_id2emb = {}
        atc3_id2emb = {}
        from resources.graphcare.get_emb import embedding_retriever
        for key in ccscm_id2name.keys():
            emb = embedding_retriever(term=ccscm_id2name[key])
            ccscm_id2emb[key] = emb

        for key in ccsproc_id2name.keys():
            emb = embedding_retriever(term=ccsproc_id2name[key])
            ccsproc_id2emb[key] = emb

        for key in atc3_id2name.keys():
            emb = embedding_retriever(term=atc3_id2name[key])
            atc3_id2emb[key] = emb
            
        ccscm_id2clus = {}
        ccsproc_id2clus = {}
        atc3_id2clus = {}

        for key in ccscm_id2emb.keys():
            emb = ccscm_id2emb[key]
            emb = np.array(emb)
            max_sim = 0
            max_id = None
            for i in range(ent_emb.shape[0]):
                emb_compare = ent_emb[i]
                sim = cosine_similarity(emb, emb_compare)
                if sim > max_sim:
                    max_sim = sim
                    max_id = i
            
            cluster_id = map_cluster_inv[str(max_id)]
            ccscm_id2clus[key] = cluster_id

        for key in ccsproc_id2emb.keys():
            emb = ccsproc_id2emb[key]
            emb = np.array(emb)
            max_sim = 0
            max_id = None
            for i in range(ent_emb.shape[0]):
                emb_compare = ent_emb[i]
                sim = cosine_similarity(emb, emb_compare)
                if sim > max_sim:
                    max_sim = sim
                    max_id = i
            
            cluster_id = map_cluster_inv[str(max_id)]
            ccsproc_id2clus[key] = cluster_id

        for key in atc3_id2emb.keys():
            emb = atc3_id2emb[key]
            emb = np.array(emb)
            max_sim = 0
            max_id = None
            for i in range(ent_emb.shape[0]):
                emb_compare = ent_emb[i]
                sim = cosine_similarity(emb, emb_compare)
                if sim > max_sim:
                    max_sim = sim
                    max_id = i
            
            cluster_id = map_cluster_inv[str(max_id)]
            atc3_id2clus[key] = cluster_id
        
        with open(f'{path}/ccscm_id2clus.json', 'w') as f:
            json.dump(ccscm_id2clus, f)
        with open(f'{path}/ccsproc_id2clus.json', 'w') as f:
            json.dump(ccsproc_id2clus, f)
        with open(f'{path}/atc3_id2clus.json', 'w') as f:
            json.dump(atc3_id2clus, f)
        
    return ccscm_id2clus, ccsproc_id2clus, atc3_id2clus, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel


def process_graph(dataset, sample_dataset, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel):
    if os.path.exists(f'resources/graphcare/{dataset}_graph.pkl'):
        with open(f'resources/graphcare/{dataset}_graph.pkl', 'rb') as f:
            G_tg = pickle.load(f)
    else:
        G = nx.Graph()

        for cluster_label, item in map_cluster.items():
            G.add_nodes_from([
                (int(cluster_label), {'y': int(cluster_label), 'x': item['embedding'][0]})
            ])

        for patient in sample_dataset:
            triple_set = set()
            conditions = flatten(patient['conditions'])
            for condition in conditions:
                cond_file = f'resources/graphcare/condition/CCSCM/{condition}.txt'
                with open(cond_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    # in case the map and emb is not up-to-date
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            r = rel2id[r]
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                                G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                            triple_set.add(triple)
                    except:
                        continue
            
            procedures = flatten(patient['procedures'])
            for procedure in procedures:
                proc_file = f'resources/graphcare/procedure/CCSPROC/{procedure}.txt'
                with open(proc_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            r = rel2id[r]
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                                G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                                triple_set.add(triple)   
                    except:
                        continue

            drugs = flatten(patient['drugs'])
            for drug in drugs:
                drug_file = f'resources/graphcare/drug/ATC3/{drug}.txt'

                with open(drug_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            r = rel2id[r]
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                edge = (int(map_cluster_inv[h]), int(map_cluster_inv[t]))
                                G.add_edge(*edge, relation=int(map_cluster_inv_rel[r]))
                                triple_set.add(triple)
                    except:
                        continue
        G_tg = from_networkx(G)
        with open(f'resources/graphcare/{dataset}_graph.pkl', 'wb') as f:
            pickle.dump(G_tg, f)

    return G_tg


def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def pad_and_convert(visits, max_visits, max_nodes):
    padded_visits = []
    for idx in range(len(visits)-1, -1, -1):
        visit_multi_hot = torch.zeros(max_nodes)
        for idx, med_code in enumerate(visits[idx]):
            visit_multi_hot[med_code] = 1
        padded_visits.append(visit_multi_hot)
    while len(padded_visits) < max_visits:
        padded_visits.append(torch.zeros(max_nodes))
    return torch.stack(padded_visits, dim=0)


def process_sample_dataset(sample_dataset, G_tg, ent2id, rel2id, map_cluster, map_cluster_inv, map_cluster_rel, map_cluster_inv_rel):
    c_v = []
    for patient in sample_dataset:
        c_v.append(len(patient['conditions']))

    max_visits = max(c_v)      

    for patient in sample_dataset:
        node_set_all = set()
        node_set_list = []
        for visit_i in range(len(patient['conditions'])):
            triple_set = set()
            node_set = set() 
            conditions = patient['conditions'][visit_i]
            procedures = patient['procedures'][visit_i]
            drugs = patient['drugs'][visit_i]

            for condition in conditions:
                cond_file = f'resources/graphcare/condition/CCSCM/{condition}.txt'
                with open(cond_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue

            for procedure in procedures:
                proc_file = f'resources/graphcare/procedure/CCSPROC/{procedure}.txt'
                with open(proc_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue

            for drug in drugs:
                drug_file = f'resources/graphcare/drug/ATC3/{drug}.txt'

                with open(drug_file, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    try:
                        items = line.split('\t')
                        if len(items) == 3:
                            h, r, t = items
                            t = t[:-1]
                            h = ent2id[h]
                            # r = int(rel2id[r]) + len(ent_emb)
                            t = ent2id[t]
                            triple = (h, r, t)
                            if triple not in triple_set:
                                triple_set.add(triple)
                                node_set.add(int(map_cluster_inv[h]))
                                # node_set.add(r)
                                node_set.add(int(map_cluster_inv[t]))
                    except:
                        continue

            node_set_list.append([*node_set])
            node_set_all.update(node_set)

        padded_visits = pad_and_convert(node_set_list, max_visits, len(G_tg.x))
        patient['node_set'] = [*node_set_all]
        patient['visit_padded_node'] = padded_visits
    return sample_dataset


def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def label_ehr_nodes(sample_dataset, max_nodes, ccscm_id2clus, ccsproc_id2clus, atc3_id2clus):
    for patient in sample_dataset:
        nodes = []
        for condition in flatten(patient['conditions']):
            ehr_node = ccscm_id2clus[condition]
            nodes.append(int(ehr_node))
            patient['node_set'].append(int(ehr_node))

        for procedure in flatten(patient['procedures']):
            ehr_node = ccsproc_id2clus[procedure]
            nodes.append(int(ehr_node))
            patient['node_set'].append(int(ehr_node))

        for drug in flatten(patient['drugs']):
            ehr_node = atc3_id2clus[drug]
            nodes.append(int(ehr_node))
            patient['node_set'].append(int(ehr_node))

        # make one-hot encoding
        node_vec = np.zeros(max_nodes)
        node_vec[nodes] = 1
        
        patient['ehr_node_set'] = torch.tensor(node_vec)

    return sample_dataset


def get_rel_emb(map_cluster_rel):
    rel_emb = []

    for i in range(len(map_cluster_rel.keys())):
        rel_emb.append(map_cluster_rel[str(i)]['embedding'][0])

    rel_emb = np.array(rel_emb)
    return torch.tensor(rel_emb)


def get_subgraph(G, dataset, idx):
    patient = dataset[idx]
    while len(patient['node_set']) == 0:
        idx -= 1
        patient = dataset[idx]

    _, _, _, edge_mask = k_hop_subgraph(torch.tensor(patient['node_set']), 1, G.edge_index)
    mask_idx = torch.where(edge_mask)[0]
    L = G.edge_subgraph(mask_idx)
    P = L.subgraph(torch.tensor(patient['node_set']))
    
    P.visit_node = patient['visit_padded_node'].float() 
    P.ehr_nodes = patient['ehr_node_set'].float()
    P.label = patient['label']
    if "text" in patient:
        P.text = patient["text"]
    if "perplexity" in patient:
        P.perplexity = patient["perplexity"]
    if "embedding" in patient:
        P.embedding = patient["embedding"]
    return {"data": P}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, G, dataset):
        self.G = G
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return get_subgraph(G=self.G, dataset=self.dataset, idx=idx)


class BiAttentionGNNConv(MessagePassing):
    def __init__(self, nn: torch.nn.Module, eps: float = 0.,
                 train_eps: bool = False, edge_dim: Optional[int] = None,
                 edge_attn=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.edge_attn = edge_attn
        if edge_attn:
            # self.W_R = torch.nn.Linear(edge_dim, edge_dim)
            self.W_R = torch.nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None, attn: Tensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, attn=attn)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        if self.W_R is not None:
            w_rel = self.W_R(edge_attr)
        else:
            w_rel = None

        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:

        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            out = (x_j * attn + w_rel * edge_attr).relu()
        else:
            out = (x_j * attn).relu()
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


class GraphCare(nn.Module):
    def __init__(
            self, dataset, mode, feature_keys, label_key, num_nodes, num_rels, max_visit, embedding_dim, hidden_dim, 
            layers=3, dropout=0.5, decay_rate=0.03, node_emb=None, rel_emb=None,
            freeze=False, patient_mode="joint", use_alpha=True, use_beta=True, use_edge_attn=True, 
            self_attn=0., gnn="BAT", attn_init=None
        ):
        super().__init__()
        self.dataset = dataset
        self.mode = mode
        self.feature_keys = feature_keys
        self.label_key = label_key
        self.gnn = gnn
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
        self.patient_mode = patient_mode
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.edge_attn = use_edge_attn
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.max_visit = max_visit
        self.label_tokenizer = self.get_label_tokenizer()
        out_channels = self.get_output_size(self.label_tokenizer)
        
        j = torch.arange(max_visit).float()
        self.lambda_j = nn.Parameter(torch.exp(self.decay_rate * (max_visit - j)).unsqueeze(0).reshape(1, max_visit, 1).float(), requires_grad=False)

        if node_emb is None:
            self.node_emb = nn.Embedding(num_nodes, embedding_dim)
        else:
            self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=freeze)

        if rel_emb is None:
            self.rel_emb = nn.Embedding(num_rels, embedding_dim)
        else:
            self.rel_emb = nn.Embedding.from_pretrained(rel_emb, freeze=freeze)

        self.lin = nn.Linear(embedding_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.layers = layers
        self.dropout = dropout

        self.alpha_attn = nn.ModuleDict()
        self.beta_attn = nn.ModuleDict()
        self.conv = nn.ModuleDict()
        self.bn_gnn = nn.ModuleDict()

        self.leakyrelu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.tahh = nn.Tanh()

        for layer in range(1, layers+1):
            if self.use_alpha:
                self.alpha_attn[str(layer)] = nn.Linear(num_nodes, num_nodes)

                if attn_init is not None:
                    attn_init = attn_init.float()  # Convert attn_init to float
                    attn_init_matrix = torch.eye(num_nodes).float() * attn_init  # Multiply the identity matrix by attn_init
                    self.alpha_attn[str(layer)].weight.data.copy_(attn_init_matrix)  # Copy the modified attn_init_matrix to the weights

                else:
                    nn.init.xavier_normal_(self.alpha_attn[str(layer)].weight)
            if self.use_beta:
                self.beta_attn[str(layer)] = nn.Linear(num_nodes, 1)
                nn.init.xavier_normal_(self.beta_attn[str(layer)].weight)
            if self.gnn == "BAT":
                self.conv[str(layer)] = BiAttentionGNNConv(nn.Linear(hidden_dim, hidden_dim), edge_dim=hidden_dim, edge_attn=self.edge_attn, eps=self_attn)
            elif self.gnn == "GAT":
                self.conv[str(layer)] = GATConv(hidden_dim, hidden_dim)
            elif self.gnn == "GIN":
                self.conv[str(layer)] = GINConv(nn.Linear(hidden_dim, hidden_dim))

            # self.bn_gnn[str(layer)] = nn.BatchNorm1d(hidden_dim)

        if self.patient_mode == "joint":
            self.MLP = nn.Linear(hidden_dim * 2, out_channels)
        else:
            self.MLP = nn.Linear(hidden_dim, out_channels)

    def get_label_tokenizer(self, special_tokens=None) -> Tokenizer:
        """Gets the default label tokenizers using `self.label_key`.

        Args:
            special_tokens: a list of special tokens to add to the tokenizer.
                Default is empty list.

        Returns:
            label_tokenizer: the label tokenizer.
        """
        if special_tokens is None:
            special_tokens = []
        label_tokenizer = Tokenizer(
            self.dataset.get_all_tokens(key=self.label_key),
            special_tokens=special_tokens,
        )
        return label_tokenizer

    def get_output_size(self, label_tokenizer: Tokenizer) -> int:
        """Gets the default output size using the label tokenizer and `self.mode`.

        If the mode is "binary", the output size is 1. If the mode is "multiclass"
        or "multilabel", the output size is the number of classes or labels.

        Args:
            label_tokenizer: the label tokenizer.

        Returns:
            output_size: the output size of the model.
        """
        output_size = label_tokenizer.get_vocabulary_size()
        if self.mode == "binary":
            assert output_size == 2
            output_size = 1
        return output_size

    def prepare_labels(
        self,
        labels: Union[List[str], List[List[str]]],
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
        """Prepares the labels for model training and evaluation.

        This function converts the labels to different formats depending on the
        mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1)
            - multiclass: a tensor of shape (batch_size,)
            - multilabel: a tensor of shape (batch_size, num_labels)

        Args:
            labels: the raw labels from the samples. It should be
                - a list of str for binary and multiclass classificationa
                - a list of list of str for multilabel classification
            label_tokenizer: the label tokenizer.

        Returns:
            labels: the processed labels.
        """
        if self.mode in ["binary"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.FloatTensor(labels).unsqueeze(-1)
        elif self.mode in ["multiclass"]:
            labels = label_tokenizer.convert_tokens_to_indices(labels)
            labels = torch.LongTensor(labels)
        elif self.mode in ["multilabel"]:
            # convert to indices
            labels_index = label_tokenizer.batch_encode_2d(
                labels, padding=False, truncation=False
            )
            # convert to multihot
            num_labels = label_tokenizer.get_vocabulary_size()
            labels = batch_to_multihot(labels_index, num_labels)
        else:
            raise NotImplementedError
        labels = labels.cuda()
        return labels

    def get_loss_function(self) -> Callable:
        """Gets the default loss function using `self.mode`.

        The default loss functions are:
            - binary: `F.binary_cross_entropy_with_logits`
            - multiclass: `F.cross_entropy`
            - multilabel: `F.binary_cross_entropy_with_logits`

        Returns:
            The default loss function.
        """
        if self.mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "multiclass":
            return F.cross_entropy
        elif self.mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "regression":
            return F.mse_loss
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Prepares the predicted probabilities for model evaluation.

        This function converts the predicted logits to predicted probabilities
        depending on the mode. The default formats are:
            - binary: a tensor of shape (batch_size, 1) with values in [0, 1],
                which is obtained with `torch.sigmoid()`
            - multiclass: a tensor of shape (batch_size, num_classes) with
                values in [0, 1] and sum to 1, which is obtained with
                `torch.softmax()`
            - multilabel: a tensor of shape (batch_size, num_labels) with values
                in [0, 1], which is obtained with `torch.sigmoid()`

        Args:
            logits: the predicted logit tensor.

        Returns:
            y_prob: the predicted probability tensor.
        """
        if self.mode in ["binary"]:
            y_prob = torch.sigmoid(logits)
        elif self.mode in ["multiclass"]:
            y_prob = F.softmax(logits, dim=-1)
        elif self.mode in ["multilabel"]:
            y_prob = torch.sigmoid(logits)
        else:
            raise NotImplementedError
        return y_prob

    def forward(self, data, **kwargs):
        data = data.cuda()
        node_ids = data.y#.cuda()
        rel_ids = data.relation#.cuda()
        edge_index = data.edge_index#.cuda()
        batch = data.batch#.cuda()
        label = data.label.tolist()
        visit_node = data.visit_node.reshape(len(label), -1, data.visit_node.shape[1]).float()#.cuda()
        ehr_nodes = data.ehr_nodes.reshape(len(label), -1).float()#.cuda()
        
        x = self.node_emb(node_ids).float()
        edge_attr = self.rel_emb(rel_ids).float()

        x = self.lin(x)
        edge_attr = self.lin(edge_attr)

        for layer in range(1, self.layers + 1):
            if self.use_alpha:
                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())), dim=1)  # (batch, max_visit, num_nodes)

            if self.use_beta:
                beta = torch.tanh((self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j

            if self.use_alpha and self.use_beta:
                attn = alpha * beta
            elif self.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, self.max_visit, 1), device='cuda')
            elif self.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes), device='cuda')
            else:
                attn = torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes), device='cuda')
                
            attn = torch.sum(attn, dim=1)
            
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)
            else:
                x = self.conv[str(layer)](x, edge_index)
            
            # x = self.bn_gnn[str(layer)](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        if self.patient_mode == "joint" or self.patient_mode == "graph":
            # patient graph embedding through global mean pooling
            x_graph = global_mean_pool(x, batch)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)

        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i]) for i in range(batch.max().item() + 1)])
            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)

        if self.patient_mode == "joint":
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
            logits = self.MLP(x_concat)
        elif self.patient_mode == "graph":
            logits = self.MLP(x_graph)
        elif self.patient_mode == "node":
            logits = self.MLP(x_node)

        y_true = self.prepare_labels(label, self.label_tokenizer)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = x_concat
        return results
