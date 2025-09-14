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
import scipy as sp
from scipy.spatial import distance 
SAVE_PATH = "results"
MIN_TOPICS = 2
MAX_TOPICS = 30
BEST_K = 6

def get_clusters():
    approach = ['wo-past']
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
                
                # Save doc-topic distr
                save_path = f'{root_dir}/{decade}/jsd_doc2doc.txt'
                u.save_dense_matrix(jsd_matrix, save_path)
    
    logger.debug('Getting doc2doc matrices complete.')
                
def get_jds_between_docs(doc_topic_distr):
    logger.debug(f'Shape of doc-topic distribution: {doc_topic_distr.shape}')
    
    jsd_matrix = distance.pdist(doc_topic_distr, metric='jensenshannon')
    jsd_matrix = distance.squareform(jsd_matrix)
    
    logger.debug(f'JS Distance matrix shape: {jsd_matrix.shape}')
    
    return jsd_matrix