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
#from igraph import arpack_options
SAVE_PATH = "results"
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
                
                pd.DataFrame({'num_communities': [comm_len],
                              'modularity': [comm_mod],
                              'lattice_number': [lattice_number],
                              'optimal_percolation': [opt_per], 
                              'num_nodes': num_nodes}).to_csv(f'{save_path}/community_metadata.csv', index=False)
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