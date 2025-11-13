# Internal
from src import utils as u
from src.tasks.linking.model import Model
# Reporting
import random
random.seed(42)
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
# File manipulation
import os
# Tensor manipulation
import pandas as pd
import numpy as np
np.random.seed(42)
np.set_printoptions(precision=5)
from igraph import *
# Visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# PyTorch
import torch
from torch import Tensor
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader

# def trainer():
#     # Load features
#     doc_topic_distr = 'results/wo-past/1800/topics_2_30/doc_topic_distr_6.txt'
#     doc_author_file = 'results/wo-past/1800/kg/doc_author_matrix.txt'

#     doc_topic = u.load_dense_matrix(doc_topic_distr) # shape: [num_docs, num_topics]
#     doc_author = np.loadtxt(doc_author_file)  # shape: [num_docs, num_authors]
    
#     num_docs = doc_topic.shape[0]
#     num_authors = doc_author.shape[1]

#     # Doc-author edges
#     doc_author_edge_index = np.array(np.nonzero(doc_author))  # shape: [2, num_edges]

#     # Doc-doc edges
#     edge_index_file = 'results/wo-past/1800/kg/edge_index.csv'
#     doc_doc_edge_index = np.loadtxt(edge_index_file, delimiter=',', skiprows=1).T  # shape: [2, num_edges]

#     data = HeteroData()
#     data['doc'].x = torch.tensor(doc_topic, dtype=torch.float)
#     data['doc', 'written_by', 'author'].edge_index = torch.tensor(doc_author_edge_index, dtype=torch.long)
#     data['doc', 'related_to', 'doc'].edge_index = torch.tensor(doc_doc_edge_index, dtype=torch.long)
#     data['author'].node_id = torch.arange(num_authors)
    
#     # Ensure undirected edges
#     data = T.ToUndirected()(data)
    
#     # Get training (80%), validation (10%) and test (10%) edges
#     train_loader, val_data, test_data = data_sampling(data)
    
#     # Instantiate model
#     model = Model(hidden_channels=64, num_doc_features=doc_topic.shape[1], num_authors=num_authors)
#     model = to_hetero(model, data.metadata(), aggr='sum') #NOTE: this is important to handle heterogeneous data
    
#     # Training loop
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logger.info(f"Device: '{device}'")

#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#     for epoch in range(1, 2):
#         total_loss = total_examples = 0
#         for sampled_data in tqdm(train_loader):
#             optimizer.zero_grad()

#             sampled_data = sampled_data.to(device)
#             pred = model(sampled_data)

#             ground_truth = sampled_data["doc", "written_by", "author"].edge_label
#             loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

#             loss.backward()
#             optimizer.step()
#             total_loss += float(loss) * pred.numel()
#             total_examples += pred.numel()
#         logger.info(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
def trainer():
    # Load features
    doc_topic_distr = 'results/wo-past/1800/topics_2_30/doc_topic_distr_6.txt'
    doc_author_file = 'results/wo-past/1800/kg/doc_author_matrix.txt'

    doc_topic = u.load_dense_matrix(doc_topic_distr)  # shape: [num_docs, num_topics]
    doc_author = np.loadtxt(doc_author_file)  # shape: [num_docs, num_authors]
    
    num_docs = doc_topic.shape[0]
    num_authors = doc_author.shape[1]

    # Doc-author edges
    doc_author_edge_index = np.array(np.nonzero(doc_author))  # shape: [2, num_edges]

    # Doc-doc edges
    edge_index_file = 'results/wo-past/1800/kg/edge_index.csv'
    doc_doc_edge_index = np.loadtxt(edge_index_file, delimiter=',', skiprows=1).T  # shape: [2, num_edges]

    data = HeteroData()
    data['doc'].x = torch.tensor(doc_topic, dtype=torch.float)
    data['doc', 'written_by', 'author'].edge_index = torch.tensor(doc_author_edge_index, dtype=torch.long)
    data['doc', 'related_to', 'doc'].edge_index = torch.tensor(doc_doc_edge_index, dtype=torch.long)
    data['author'].node_id = torch.arange(num_authors)
    
    # Ensure undirected edges
    data = T.ToUndirected()(data)
    
    # Get training (80%), validation (10%) and test (10%) edges
    train_loader, val_data, test_data = data_sampling(data)
    
    # Instantiate model
    model = Model(hidden_channels=64, num_doc_features=doc_topic.shape[1], num_authors=num_authors)
    model = to_hetero(model, data.metadata(), aggr='sum')  # NOTE: this is important to handle heterogeneous data
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: '{device}'")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 2):
        total_loss = total_examples = 0
        for sampled_data in tqdm(train_loader):
            optimizer.zero_grad()

            # Debug sampled_data
            print(sampled_data)

            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)

            ground_truth = sampled_data["doc", "written_by", "author"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)

            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        logger.info(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    
# def data_sampling(data):
#     transform = T.RandomLinkSplit(
#     num_val=0.1,
#     num_test=0.1,
#     disjoint_train_ratio=0.3,
#     neg_sampling_ratio=2.0,
#     add_negative_train_samples=True, #NOTE: set to False if you want to sample negative edges during training
#     edge_types=("doc", "written_by", "author"),
#     rev_edge_types=("doc", "related_to", "doc"), 
#     )

#     train_data, val_data, test_data = transform(data)

#     # Define seed edges for your edge type:
#     edge_label_index = train_data["doc", "written_by", "author"].edge_label_index
#     edge_label = train_data["doc", "written_by", "author"].edge_label

#     train_loader = LinkNeighborLoader(
#         data=train_data,
#         num_neighbors=[20, 10],
#         neg_sampling_ratio=2.0,
#         edge_label_index=(("doc", "written_by", "author"), edge_label_index),
#         edge_label=edge_label,
#         batch_size=128,
#         shuffle=True,
#     )
    
#     return train_loader, val_data, test_data

def data_sampling(data):
    transform = T.RandomLinkSplit(
        num_val=0.05,  # Less aggressive split
        num_test=0.05,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=True,
        edge_types=("doc", "written_by", "author"),
        rev_edge_types=("doc", "related_to", "doc"),
    )

    train_data, val_data, test_data = transform(data)

    # Debug checks
    print("Edge types in train_data:", train_data.edge_types)
    if train_data["doc", "written_by", "author"] is None:
        raise ValueError("No 'doc'-'written_by'-'author' edges in training data after split!")

    if train_data["doc", "written_by", "author"].edge_label_index is None:
        raise ValueError("edge_label_index is missing for 'doc'-'written_by'-'author'!")

    # Define seed edges for your edge type:
    edge_label_index = train_data["doc", "written_by", "author"].edge_label_index
    edge_label = train_data["doc", "written_by", "author"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("doc", "written_by", "author"), edge_label_index),
        edge_label=edge_label,
        batch_size=128,
        shuffle=True,
    )
    
    return train_loader, val_data, test_data    