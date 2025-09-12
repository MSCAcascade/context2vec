# Internal
from src.tasks.classification.eval import models_evaluation
from src import utils as u
from src.tasks.classification import model
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
import scipy as sp
from scipy.spatial import distance #REVIEW: this is JS Distance, the sqrt(JSD)
from sklearn.metrics.pairwise import cosine_similarity
# Topic modeling
from tmtoolkit.topicmod.model_stats import generate_topic_labels_from_top_words
from tmtoolkit.bow.bow_stats import doc_lengths
from tmtoolkit.topicmod.model_io import ldamodel_top_topic_words
from tmtoolkit.topicmod.visualize import generate_wordclouds_for_topic_words
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import plotly.express as px
BEST_K = 6
SAVE_PATH = 'results'

<<<<<<< HEAD
APP = 'wo-past'

KEY_TOPICS_WPAST = {'1750': 'water', '1760': 'water', '1770': 'water', '1780': 'air', '1790': 'air', '1800': 'air'}
KEY_TOPICS_WOPAST = {'1760': 'water', '1770': 'air', '1780': 'air', '1790': 'acid', '1800': 'acid'}
=======
APP = 'w-past'

KEY_TOPICS_WPAST = {'1760': 'wind', '1770': 'wind', '1780': 'air', '1790': 'air', '1800': 'air'}
KEY_TOPICS_WOPAST = {'1760': 'wind', '1770': 'air', '1780': 'air', '1790': 'acid', '1800': 'acid'}
>>>>>>> 7e347b6 (saved results)

def get_key_topics():
    # Remove the approach loop and use only wo-past
    key_topics = KEY_TOPICS_WOPAST
    decades = ['1760', '1770', '1780', '1790', '1800']
    
    # Store aggregated statistics instead of raw distributions
    topic_stats = []
    topic_labels = []
    
    root_dir = f'{SAVE_PATH}/{APP}'
    decade_dirs = os.listdir(root_dir) 
    decade_dirs.sort()
    
    for decade in tqdm(decade_dirs, desc='Processing  decades'):
        if decade in decades:
            # Load data
            doc_topic_distr = u.load_dense_matrix(f'{root_dir}/{decade}/topics_2_30/doc_topic_distr_{BEST_K}.txt')
            topic_labels_dict = u.load_object(f'{root_dir}/{decade}/topics_2_30/topic_labels_{BEST_K}.pkl')
            topic_labels_dict = {k: v.split('_')[1] if '_' in v else v for k, v in topic_labels_dict.items()}
            
            # Find the index of the key topic
            try: 
                key_topic = key_topics[decade]
                key_topic = key_topic.split('_')[0] if '_' in key_topic else key_topic
                key_topic_index = list(topic_labels_dict.values()).index(key_topic)
            except ValueError:
                logger.error(f'Key topic "{key_topic}" not found in topic labels for decade {decade}. Skipping.')
                continue
            
            # Get the topic distribution for the key topic
            key_topic_distribution = doc_topic_distr[:, key_topic_index]
            
            # Create summary statistics (same length for all)
            stats = [
                np.mean(key_topic_distribution),
                np.std(key_topic_distribution),
                np.median(key_topic_distribution),
                np.percentile(key_topic_distribution, 25),
                np.percentile(key_topic_distribution, 75),
                np.min(key_topic_distribution),
                np.max(key_topic_distribution)
            ]
            
            topic_stats.append(stats)
            topic_labels.append(f'{key_topic}_{decade}')

    
    # Convert to numpy array - all have same length (7 statistics)
    topic_stats = np.array(topic_stats)
    logger.info(f'Topic statistics shape: {topic_stats.shape}')
    
    # Normalize to create probability-like distributions
    topic_stats_norm = topic_stats / topic_stats.sum(axis=1, keepdims=True)
    
    # Compute Jensen-Shannon distance
    js_distance_matrix = distance.pdist(topic_stats_norm, metric='jensenshannon')
    js_distance_matrix = distance.squareform(js_distance_matrix)
    
    # Create DataFrame
    js_distance_df = pd.DataFrame(js_distance_matrix, 
                                  index=topic_labels, 
                                  columns=topic_labels)
    
    # Create heatmap
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(js_distance_df, annot=True, cmap='coolwarm', fmt='.1f', square=True)
    # plt.title('Jensen-Shannon Distance Matrix for Key Topics\n')
    # plt.xticks(rotation=45, ha='right')
    # plt.yticks(rotation=0)
    # plt.tight_layout()
    # plt.savefig(f'{SAVE_PATH}/jsd_matrix_wopast.png', dpi=300, bbox_inches='tight')
    
        # Create a clustermap with the row dendrogram
    row_linkage = linkage(js_distance_matrix, method='average')

    cmap = sns.light_palette("blue", reverse=True, as_cmap=True)
    sns.set_theme(font_scale=1.4)
    
    # Create clean labels without years for display
    clean_labels = [label.split('_')[0] for label in topic_labels]
    
    
    g = sns.clustermap(js_distance_matrix,
                        row_cluster=False,  #NOTE:To disable row clustering
                        col_cluster=False,  
                        cmap=cmap,
                        vmin=0, 
                        vmax=0.2,  # JSD is bounded between 0 and 1
                        xticklabels=clean_labels,
                        yticklabels=clean_labels, 
                        annot=True, 
                        fmt=".2f", 
                        annot_kws={"size": 16},
                        cbar_kws={'label': 'JS distance', 'ticks': [0,0.1,0.2]})

    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=16)  # Rotate y-tick labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90,fontsize=16)  # Rotate x-tick labels
    
    # Alternative: Add as axis labels
    g.ax_heatmap.set_xlabel('1760 ------------------------------------------------------------------------------------------------------> 1800', fontsize=14)
    g.ax_heatmap.set_ylabel('<---------------------------------------------------------------------------------------------------------1760', fontsize=14)
    
    plt.savefig(f'{SAVE_PATH}/jsd_heatmap_{APP}.png', bbox_inches='tight', dpi=300)
    
    logger.info('Key topics analysis complete. Heatmap saved.')


