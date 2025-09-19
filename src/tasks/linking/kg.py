# Internal
from src import utils as u
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

SAVE_PATH = "results"

def get_acid_features(acid_authors_1790, acid_authors_1800):
    # Load txt files
    acid_1790_df = pd.read_csv(acid_authors_1790, sep='\t', header=None, names=['text_id', 'author'])
    acid_1800_df = pd.read_csv(acid_authors_1800, sep='\t', header=None, names=['text_id', 'author'])
    # Process author data
    acid_1790_df = get_clean_authors(acid_1790_df)
    acid_1800_df = get_clean_authors(acid_1800_df)
    # Ensure authors data -> list 
    acid_1790_authors = acid_1790_df['author'].tolist()
    acid_1800_authors = acid_1800_df['author'].tolist()
    # Find unique authors
    total_authors = list(set(acid_1790_authors + acid_1800_authors))
    # Plot authors in pie chart
    author_counts_1790 = acid_1790_df['author'].value_counts()
    author_counts_1800 = acid_1800_df['author'].value_counts()
    # Combine counts into a single DataFrame
    combined_counts = pd.DataFrame({
        '1790': author_counts_1790,
        '1800': author_counts_1800
    }).fillna(0)
    combined_counts = combined_counts.astype(int)
    combined_counts['total'] = combined_counts['1790'] + combined_counts['1800']
    combined_counts = combined_counts.sort_values(by='total', ascending=False)
    top10_counts = combined_counts.head(10).drop(columns='total') 

    # Plotting
    plt.figure(figsize=(10, 8))
    top10_counts.plot(kind='bar', stacked=True)
    plt.title('Top-10 authors in "acid" topic')
    plt.ylabel('papers')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='decade')
    plt.tight_layout()
    # Save plot
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plot_path = os.path.join(results_folder, 'author_distribution_top10_1790_1800.png')
    plt.savefig(plot_path)
    logger.debug(f'Plot saved to {plot_path}')
    plt.close()
    
    
def get_clean_authors(df):
    # Get the first author for each author in the authors column # TODO: later implement logic to account for multiple authors
    df["author"] = df["author"].str.split("|").str[0].str.strip()
    return df

def get_wc():
    from wordcloud import WordCloud
    from PIL import Image
    approach = ['w-past'] #TODO: add this to config file
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    
    for app in tqdm(approach, desc='Getting percentiles'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort() #NOTE: 1750->1800
        
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                # Load topic words
                filename = f'{root_dir}/{decade}/topics_2_30/topics_top_50_words.csv'
                topic_word_df = pd.read_csv(filename)
                # transpose df 
                topic_word_df = topic_word_df.T
                topic_word_df.columns = topic_word_df.iloc[0]
                topic_word_df = topic_word_df.drop(topic_word_df.index[0])
                
                save_path = f'{root_dir}/{decade}/wordclouds'
                os.makedirs(save_path, exist_ok=True)
                for col in topic_word_df.columns:
                    # Striṕ the cell value in two given the space
                    matrix = topic_word_df[col].str.split(' ', expand=True)
                    # Get words and probabilities
                    words = list(matrix.iloc[:,0])
                    probability = list(matrix.iloc[:,1])
                    probability = [p.replace('(','').replace(')','') for p in probability]
                    # Ṕrocess data
                    dic = {'words':words,'probability':probability}
                    df = pd.DataFrame(data=dic)
                    df[["probability"]] = df[["probability"]].apply(pd.to_numeric)
                    df = df.astype({"words": str, "probability": float})
                    data = df.set_index('words').to_dict()['probability']

                    # Create wordcloud
                    mask = np.array(Image.open('images/brain.png')) #TODO: maybe change it later
                    wc = WordCloud( max_words=1000, background_color="white", colormap='winter',
                    mask=mask, max_font_size=256, width=mask.shape[1],
                    height=mask.shape[0]).generate_from_frequencies(data)
                    
                    # Plot
                    my_dpi = 92
                    fig = plt.figure(figsize=(1000/my_dpi,1000/my_dpi),dpi=my_dpi)

                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')

                    # Remove what's before the underscore in col
                    col = col.split('_')[1] if '_' in col else col
                    fig.savefig(f'{save_path}/wc_{col}',dpi=my_dpi * 10, bbox_inches='tight')
                    plt.close()
def get_percentiles():
    approach = ['wo-past'] #TODO: add this to config file
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    
    for app in tqdm(approach, desc='Getting percentiles'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort() #NOTE: 1750->1800
        
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                # Load jsd matrix
                filename = f'{root_dir}/{decade}/jsd_doc2doc.txt'
                jsd_matrix = u.load_dense_matrix(filename)
    
                jsd_matrix[np.isnan(jsd_matrix)] = 0 #NOTE: set NaN to 0 (happens when comparing identical docs) 
                
                # Get thresholds
                Threshold=[]
                for x in range(1,80,1): # NOTE: Distribution percentiles (1-80) to test different thresholds for graph creation 
                    Threshold.append(np.percentile(jsd_matrix,x))

                size= len(jsd_matrix)
                dict_data = {}
                for t in Threshold:
                    Adj_matrix = np.zeros(np.shape(jsd_matrix))
                    for i in range(0,size):
                        for j in range(0,size):
                            if jsd_matrix[i,j] < t:
                                Adj_matrix[i,j] = 1
                            else:
                                Adj_matrix[i,j] = 0
                    np.fill_diagonal(Adj_matrix,0)
                    dict_data[t]= Adj_matrix 

                # Get threshold 
                percolation = []
                giant_comp_list = []
                for key,value in dict_data.items(): 
                    Z = Graph.Adjacency(value.astype(bool).tolist(), mode=ADJ_UNDIRECTED) 
                    cl = Z.components()
                    lcc = cl.giant()
                    giant_comp = lcc.vcount()
                    per = lcc.vcount() / size # NOTE: percolation ratio: size of giant component / size of graph
                    # NOTE: we aim for the point of percolation transition
                    
                    giant_comp_list.append(giant_comp)
                    percolation.append(per)
                
                # Save percentiles
                percentile_df = pd.DataFrame({'giant_component': giant_comp_list,'percolation': percolation})

                save_path = f'{root_dir}/{decade}/kg'
                os.makedirs(save_path, exist_ok=True)
                
                percentile_df.to_csv(f'{save_path}/percentiles.csv', index=False)
                logger.debug(f'Saved percentiles to {save_path}/percentiles.csv')
                
                # Get percolation transition
                opt_per = calculate_percolation_transition(percentile_df)
                
                # Get text IDs
                filename = f'{root_dir}/{decade}/df_{decade}.csv'
                textId_df = pd.read_csv(filename)
                
                # Get optimal graph
                Z, lattice_number = get_opt_graph(jsd_matrix, opt_per, textId_df)
                
                # Get communities
                comm_multi, comm_len, comm_mod = get_communities(Z)
                
                # Save results
                
                ## Adj matrix
                np.savetxt(f'{save_path}/adj_matrix.txt', Adj_matrix, fmt = '%d', delimiter = ',')
                ## Graph
                save(Z, f'{save_path}/graph.gml')
                ## Comm data
                membership = comm_multi.membership
                pd.DataFrame({'membership': membership}).to_csv(f'{save_path}/community_membership.csv', index=False)
                ## Comm metadata
                num_nodes = Z.vcount()
                lcc_percent = lcc.vcount() / num_nodes  # lcc is from get_opt_graph

                pd.DataFrame({
                    'num_communities': [comm_len],
                    'modularity': [comm_mod],
                    'lattice_number': [lattice_number],
                    'optimal_percolation': [opt_per], 
                    'num_nodes': num_nodes,
                    'lcc_percent': [lcc_percent]   # <-- add this line
                }).to_csv(f'{save_path}/community_metadata.csv', index=False)

                ## Edge index
                edge_index = np.array([(e.source, e.target) for e in Z.es]).T  # shape: (2, num_edges)
                edge_names = [(Z.vs[e.source]['name'], Z.vs[e.target]['name']) for e in Z.es]
                
                pd.DataFrame(edge_index.T, columns=['source_idx', 'target_idx']).to_csv(f'{save_path}/edge_index.csv', index=False)
                pd.DataFrame(edge_names, columns=['source_name', 'target_name']).to_csv(f'{save_path}/edge_names.csv', index=False)
                
                # Plot graph
                plot_graph(Z, comm_multi, save_path)
                
def calculate_percolation_transition(percentile_df):
    percolation = percentile_df['percolation'].values

    # Compute discrete derivative
    delta = np.diff(percolation)
    transition_idx = np.argmax(delta) # NOTE: Find max slope in percolation curve
    
    opt_per = percolation[transition_idx]
    
    return opt_per

def get_opt_graph(jsd_matrix, opt_per, textId_df):
    Threshold=[np.percentile(jsd_matrix,opt_per*100)] # NOTE: Get threshold corresponding to optimal percolation point
    size= len(jsd_matrix)
    dict_data = {}
    for t in Threshold:
        Adj_matrix = np.zeros(np.shape(jsd_matrix))
        for i in range(0,size):
            for j in range(0,size):
                if jsd_matrix[i,j] < t:
                    Adj_matrix[i,j] = 1
                else:
                    Adj_matrix[i,j] = 0
        np.fill_diagonal(Adj_matrix,0)
        dict_data[t]= Adj_matrix

    percolation = []
    node_names = textId_df['text_id'].to_numpy() # NOTE: why convert to numpy?

    # Get giant component and evaluate size vs original
    for key,value in dict_data.items(): 
        Z = Graph.Adjacency(value.astype(bool).tolist(), mode=ADJ_UNDIRECTED) 
        Z.vs['name']=node_names
        cl = Z.components()
        lcc = cl.giant()
        per = lcc.vcount() / size
        
        logger.info(f'Giant component size: {lcc.vcount()} out of {size} nodes ({per*100:.2f}%)')
        
    lattice_number = Z.ecount()
    return Z, lattice_number

def get_communities(graph): #NOTE: multilevel is a version of Louvain
    """ Get communities using the Louvain method.
    Args:
        graph: An igraph graph object.
    Returns:
        communities: A list of communities.
    """
    comm_multi = graph.community_multilevel()
    comm_multi_len = len(comm_multi) 
    comm_multi_mod = comm_multi.modularity
    return comm_multi, comm_multi_len, comm_multi_mod

def plot_graph(Z, comm_multi, save_path):
    import igraph as ig
    layout = Z.layout('fr')
    pal = ig.drawing.colors.ClusterColoringPalette(len(comm_multi))
    Z.vs['color'] = pal.get_many(comm_multi.membership)
    ig.plot(Z,f'{save_path}/graph_communities.svg',layout = layout)


def get_authors_matrix():
    # # Load graph #NOTE: Uncomment to get node names
    # g_file = 'results/wo-past/1800/kg/graph.gml'
    # Z = load(g_file)
    # # Get node names
    # node_names = Z.vs['name']
    # # Save node names in txt file
    # with open('results/wo-past/1800/kg/graph_node_names.txt', 'w') as f:
    #     for name in node_names:
    #         f.write(f"{name}\n")
    # logger.debug(f'Saved node names to results/wo-past/1800/kg/graph_node_names.txt')
    # Load authors data
    filename = 'results/wo-past/1800/kg/KG_1800_authors.txt'
    authors_df = pd.read_csv(filename, sep='\t', header=None, names=['text_id', 'author'])
    authors_df = get_clean_authors(authors_df)
    authors_df = authors_df['author'].tolist()
    # Create document-author matrix
    unique_authors = list(set(authors_df))
    doc_author_matrix = np.zeros((len(authors_df), len(unique_authors)), dtype=int)
    author_index = {author: idx for idx, author in enumerate(unique_authors)}
    for doc_idx, author in enumerate(authors_df):
        if author in author_index:
            author_idx = author_index[author]
            doc_author_matrix[doc_idx, author_idx] = 1
    # Save document-author matrix
    save_path = 'results/wo-past/1800/kg'
    u.save_dense_matrix(doc_author_matrix, f'{save_path}/doc_author_matrix.txt')
    # Save author index
    with open(f'{save_path}/doc_author_matrix_columns.txt', 'w') as f:
        for idx, author in enumerate(unique_authors):
            f.write(f"{idx}\t{author}\n")

    
    
    

# GNN ref: https://colab.research.google.com/drive/1r_FWLSFf9iL0OWeHeD31d_Opt031P1Nq?usp=sharing
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


# class Model(torch.nn.Module):
#     def __init__(self, hidden_channels):
#         super().__init__()
#         # Since the dataset does not come with rich features, we also learn two
#         # embedding matrices for users and movies:
#         self.movie_lin = torch.nn.Linear(20, hidden_channels)
#         self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
#         self.movie_emb = torch.nn.Embedding(data["movie"].num_nodes, hidden_channels)

#         # Instantiate homogeneous GNN:
#         self.gnn = GNN(hidden_channels)

#         # Convert GNN model into a heterogeneous variant:
#         self.gnn = to_hetero(self.gnn, metadata=data.metadata())

#         self.classifier = Classifier()

#     def forward(self, data: HeteroData) -> Tensor:
#         x_dict = {
#           "user": self.user_emb(data["user"].node_id),
#           "movie": self.movie_lin(data["movie"].x) + self.movie_emb(data["movie"].node_id),
#         } 

#         # `x_dict` holds feature matrices of all node types
#         # `edge_index_dict` holds all edge indices of all edge types
#         x_dict = self.gnn(x_dict, data.edge_index_dict)
#         pred = self.classifier(
#             x_dict["user"],
#             x_dict["movie"],
#             data["user", "rates", "movie"].edge_label_index,
#         )

#         return pred

        
# model = Model(hidden_channels=64)

# logger.info(f'Model details: {model}')