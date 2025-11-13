from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import torch
from torch import Tensor
from torch_geometric.data import HeteroData

# class GNN(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         self.conv1 = SAGEConv(hidden_channels, hidden_channels)
#         self.conv2 = SAGEConv(hidden_channels, hidden_channels)

#     def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
#         print("edge_index:", type(edge_index), edge_index.shape, edge_index.dtype) #NOTE: Debug here
        
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
class Classifier(torch.nn.Module):
    def forward(self, x_doc: Tensor, x_author: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_doc = x_doc[edge_label_index[0]]
        edge_feat_author = x_author[edge_label_index[1]]
        return (edge_feat_doc * edge_feat_author).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, num_doc_features, num_authors):
        super().__init__()
        # Linear layer for doc features
        self.doc_lin = torch.nn.Linear(num_doc_features, hidden_channels)
        # Embedding for authors
        self.author_emb = torch.nn.Embedding(num_authors, hidden_channels)
        # GNN layers
        self.gnn = GNN(hidden_channels)
        # Classifier for link prediction
        self.classifier = Classifier()

    def forward(self, data):
        # Extract node features
        x_dict = {
            "doc": self.doc_lin(data["doc"].x),
            "author": self.author_emb(data["author"].node_id),
        }
        # Pass through GNN
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        # Predict edges for the "doc-written_by-author" relation
        pred = self.classifier(
            x_dict["doc"],
            x_dict["author"],
            data["doc", "written_by", "author"].edge_label_index,
        )
        return pred

# Example instantiation 
# model = Model(hidden_channels=64, num_doc_features=doc_topic.shape[1], num_authors=num_authors)