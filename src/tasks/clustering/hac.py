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
import scipy as sp
from scipy.spatial import distance 
# Clustering
from sklearn.cluster import AgglomerativeClustering
import umap
# Visualization
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
SAVE_PATH = "results"
MIN_TOPICS = 2
MAX_TOPICS = 30
BEST_K = 6

#TODO: refactor code so under each task folder: "main"-ordered functions, "model"-chosen model function, "hac"-selected approach, etc.

def get_clusters():
    approach = ['w-past','wo-past'] #TODO: add this to config file
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    
    for app in tqdm(approach, desc='Getting clusters'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort() #NOTE: 1750->1800
        
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                # Load doc-topic distr
                filename = f'{root_dir}/{decade}/topics_{MIN_TOPICS}_{MAX_TOPICS}/doc_topic_distr_{BEST_K}.txt'
                doc_topic_distr = u.load_dense_matrix(filename)
                
                jsd_matrix = get_jds_between_docs(doc_topic_distr)
                
                # Save JS distance matrix
                save_path = f'{root_dir}/{decade}/jsd_doc2doc.txt'
                u.save_dense_matrix(jsd_matrix, save_path)

                # Perform HAC # TODO: make params a variable
                hac = AgglomerativeClustering(n_clusters=BEST_K,
                                metric='cosine',
                                linkage='average')
                
                y_hac = hac.fit_predict(doc_topic_distr)
                logger.debug(f'Shape of HAC labels: {y_hac.shape}') #TODO: create map of cluster to topic labels?
                
                # Get topic labels dict
                filename = f'{root_dir}/{decade}/topics_{MIN_TOPICS}_{MAX_TOPICS}/topic_labels_{BEST_K}.pkl'
                topic_labels_dict = u.load_object(filename)
                
                
                # NOTE: Clean all values in topic_labels_dict by removing text before the first underscore
                topic_labels_dict = {
                    k: v.split('_', 1)[1] if '_' in v else v
                    for k, v in topic_labels_dict.items()
                }
                
                
                # Get UMAP
                save_path = f'{root_dir}/{decade}/hac'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                max_topic_for_doc = get_max_topic_for_docs(doc_topic_distr)
                topic_cluster_df = get_topic_cluster_df(max_topic_for_doc, y_hac, topic_labels_dict)
                
                topic_cluster_df = get_umap(doc_topic_distr, topic_cluster_df, save_path)
                get_umap_plot(topic_cluster_df, save_path, topic_labels_dict)
                
                # Get topic-cluster correspondence
                describe_clusters(topic_cluster_df, save_path)
                logger.debug(f'Clustering for decade {decade} complete.')
    logger.info('Completed clustering analysis for all decades.')
                
def get_jds_between_docs(doc_topic_distr):
    logger.debug(f'Shape of doc-topic distribution: {doc_topic_distr.shape}')
    
    jsd_matrix = distance.pdist(doc_topic_distr, metric='jensenshannon')
    jsd_matrix = distance.squareform(jsd_matrix)
    
    logger.debug(f'JS Distance matrix shape: {jsd_matrix.shape}')
    
    return jsd_matrix

def get_max_topic_for_docs(doc_topic_dist):
    """ Get the topic number with the highest probability for each document.
    Args:
        doc_topic_dist: A dense matrix.
    Returns:
        max_topic_for_doc: A list of topic numbers.
    """
    max_topic_for_doc = np.argmax(doc_topic_dist, axis=1)
    return max_topic_for_doc.tolist()

def get_umap(doc_topic_distr, topic_cluster_df, save_path):
    """Get UMAP visualization.
    Args:
        filename: The filename of the dense matrix.
        y_hac: The cluster labels.
    Returns:
        None
    """

    mapper = umap.UMAP(n_neighbors=10, #NOTE: more neighbors -> more clustered
                        random_state=1234,
                        metric='cosine')
    embedding = mapper.fit_transform(doc_topic_distr)

    topic_cluster_df['x'] = embedding[:,0]
    topic_cluster_df['y'] = embedding[:,1]

    topic_cluster_df.to_csv(f'{save_path}/topic_cluster_umap_df.csv', index=False)
    return topic_cluster_df

def get_umap_plot(umap_df, save_path, topic_labels_dict):
    """ Create UMAP plot with clusters only, and annotate centroids with topic label. """
    plt.figure(figsize=(12,12))
    ax = sns.scatterplot(
        data=umap_df,
        x='x',
        y='y',
        hue='cluster',
        palette='Set2',
        s=24,
        ec='black',
        legend='full'
    )

    # Compute centroids
    centroids = umap_df.groupby('cluster')[['x', 'y']].mean()
    
    # Get cluster labels dict #TODO: Review later
    cluster_label_dict = umap_df.groupby('cluster')['topic'].agg(lambda x: x.mode().iloc[0]).to_dict()
    

    # Annotate centroids with topic label from topic_labels_dict
    for cluster, (x, y) in centroids.iterrows():
        cluster_label = cluster_label_dict.get(cluster, 'Unknown')
        ax.text(x, y, cluster_label, fontsize=12, weight='bold', ha='center', va='center', color='black', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
    # Remove legend
    ax.get_legend().remove()
    
    ax.set(xlabel=None, ylabel=None)
    ax.set_aspect('equal', 'datalim')
    plt.xticks([])
    plt.yticks([])
    ax.get_figure().savefig(f'{save_path}/umap.png', bbox_inches='tight')

def get_topic_cluster_df(max_topic_for_doc, y_hac, topic_labels_dict):
    """Get a DataFrame of the topic clusters.
    Args:
        max_topic_for_doc: The topic number with the highest probability for each document.
        y_hac: The cluster labels.
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    
    topic_label_vec = [topic_labels_dict[topic] for topic in max_topic_for_doc]

    df = pd.DataFrame({'topic': topic_label_vec, 'cluster': y_hac})
    return df

def describe_clusters(df, save_path):
    # Create a new column 'topic_count' that contains the count of each topic within its cluster
    df['topic_count'] = df.groupby(['cluster', 'topic'])['topic'].transform('count')

    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        logger.info('Cluster %s: %s', cluster, cluster_df['topic'].value_counts())

    # Plot the distribution of topics in each cluster
    plt.figure(figsize=(12,8))
    sns.countplot(data=df, x='cluster', hue='topic', palette='Set2')
    max_min_tuple = {}
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        topic_max = cluster_df['topic'].value_counts().idxmax()
        logger.info('Cluster %s: Topic with the most papers: %s', cluster, topic_max)
        topic_min = cluster_df['topic'].value_counts().idxmin()
        logger.info('Cluster %s: Topic with the fewest papers: %s', cluster, topic_min)
        max_min_tuple[cluster] = {'max': topic_max, 'min': topic_min}

    plt.title('Distribution of topics in each cluster', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of papers', fontsize=12)
    plt.legend(title='Topic', title_fontsize=12, fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f'{save_path}/cluster_topic_distribution.png', bbox_inches='tight')

    # save the max and min topic for each cluster in a csv file
    # max_min_df = pd.DataFrame(max_min_tuple).T
    # max_min_df.to_csv('results/hac/max_min_topic_per_cluster.csv')
    # logger.info('Saved max and min topic for each cluster in results/hac/max_min_topic_per_cluster.csv')
    # logger.info('Saved cluster topic distribution plot in results/hac/cluster_topic_distribution.png')

def get_acid_features():
    approach = ['w-past', 'wo-past']
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    
    for app in tqdm(approach, desc='Getting clusters'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort() #NOTE: 1750->1800
        
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                # Get doc labels
                filename = f'{root_dir}/{decade}/df_{decade}.csv'
                df = pd.read_csv(filename)
                text_ids = df['text_id'].tolist()
                logger.info(f'Number of documents in corpus: {len(text_ids)}')
                # Get doc clusters
                filename = f'{root_dir}/{decade}/hac/topic_cluster_umap_df.csv'
                cluster_df = pd.read_csv(filename)
                logger.info(f'Number of documents in cluster df: {cluster_df.shape[0]}')
                # Create new df: text_ids, topic, cluster
                combined_df = pd.DataFrame({'text_id': text_ids,
                                            'topic': cluster_df['topic'],
                                            'cluster': cluster_df['cluster']})
                # Remove rows which topic isn't 'acid'
                # combined_df = combined_df[combined_df['topic'].str.contains('acid', case=False)] # TODO: later automate this
                # logger.info(f'Number of documents in combined df after filtering for "acid" topic: {combined_df.shape[0]}')
                # Save combined df
                save_path = f'{root_dir}/{decade}/hac/textId_topics.csv'
                combined_df.to_csv(save_path, index=False)
                
                
    # get doc labels