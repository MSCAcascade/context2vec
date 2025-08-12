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

KEY_TOPICS_WPAST = {'1770': 'wind', '1780': 'air', '1790': 'air', '1800': 'air'}
KEY_TOPICS_WOPAST = {'1770': 'air', '1780': 'air', '1790': 'acid', '1800': 'acid'}

def get_key_topics():
    approach = ['w-past', 'wo-past']
    decades = ['1770','1780', '1790', '1800']
    
    # Store aggregated statistics instead of raw distributions
    topic_stats = []
    topic_labels = []
    
    for app in tqdm(approach, desc='Evaluating for 2 approaches: w-past and wo-past'):
        if app == 'w-past':
            key_topics = KEY_TOPICS_WPAST
        elif app == 'wo-past':
            key_topics = KEY_TOPICS_WOPAST
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort()
        
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                # Load data
                doc_topic_distr = u.load_dense_matrix(f'{root_dir}/{decade}/topics_2_30/doc_topic_distr_{BEST_K}.txt')
                topic_labels_dict = u.load_object(f'{root_dir}/{decade}/topics_2_30/topic_labels_{BEST_K}.pkl')
                topic_labels_dict = {k: v.split('_')[1] if '_' in v else v for k, v in topic_labels_dict.items()}
                
                # Find the index of the key topic
                key_topic = key_topics[decade]
                key_topic = key_topic.split('_')[0] if '_' in key_topic else key_topic
                key_topic_index = list(topic_labels_dict.values()).index(key_topic)
                
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
                topic_labels.append(f'{key_topic}_{app}_{decade}')
                
                logger.info(f'Processed {key_topic} in {app} {decade}: {len(key_topic_distribution)} documents')
    
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
    plt.figure(figsize=(12, 12))
    sns.heatmap(js_distance_df, annot=True, cmap='coolwarm', fmt='.3f', square=True)
    plt.title('Jensen-Shannon Distance Matrix for Key Topics\n')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{SAVE_PATH}/key_topics_js_distance_matrix.png', dpi=300, bbox_inches='tight')
    
    logger.info('Key topics analysis complete. Heatmap saved.')


