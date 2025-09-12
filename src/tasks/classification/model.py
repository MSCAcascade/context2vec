# Internal
from src.tasks.classification.eval import models_evaluation
from src import utils as u
from src.tasks.specialization import entropy as ent
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

SAVE_PATH = "results"
MIN_TOPICS = 2
MAX_TOPICS = 30
BEST_K = [6,6,6,6,6,6]  # Index = decade
def get_topics():
    """ Get topics with LDA model.
    Args:
        params (dict): The parameters.
    Returns:
        None
    """
    logger.info('Starting topic models evaluation...')
    approach = ['w-past', 'wo-past']
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    index = 0
    for app in tqdm(approach, desc='Evaluating for 2 approaches: w-past and wo-past'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort() #NOTE: 1750->1800
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                results_dir = f'{root_dir}/{decade}'
                filename = f'{results_dir}/dtm_sparse.npz'
                models_evaluation(filename,results_dir)
    logger.info('Topic model evaluation complete.')

def get_topic_words():
    approach = ['w-past', 'wo-past']
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    for app in tqdm(approach, desc='Evaluating for 2 approaches: w-past and wo-past'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort() #NOTE: 1750->1800
        index = 0  # Move index initialization outside the loop
        for decade in tqdm(decade_dirs, leave=False):
            if decade in decades:
                # Load models
                results_dir = f'{root_dir}/{decade}/topics_{MIN_TOPICS}_{MAX_TOPICS}'
                filename = f'{results_dir}/eval_results_by_topics.pkl'
                eval_results_by_topics = u.load_object(filename)
                
                # Collect best model
                best_k = BEST_K[index]
                index += 1
                logger.info('Best K: %s', best_k)
                best_tm, lda_object = next((item for item in eval_results_by_topics if item[0]['n_topics'] == best_k), None)
                logger.info('Best topic model: %s', best_tm)

                lda_model = lda_object['model']
                doc_topic_distr = lda_model.doc_topic_
                topic_word_distr = lda_model.topic_word_

                u.save_dense_matrix(doc_topic_distr, f'{results_dir}/doc_topic_distr_{best_k}.txt')
                u.save_dense_matrix(topic_word_distr, f'{results_dir}/topic_word_distr_{best_k}.txt')
                logger.info('Saved document-topic and topic-word distributions in %s', results_dir)
                
                # Get topic labels
                dtm_filename = f'{root_dir}/{decade}/dtm_sparse.npz'
                vocab_filename = f'{root_dir}/{decade}/vocab.pkl'
                topic_labels_dict = get_topic_labels(dtm_filename, vocab_filename, topic_word_distr, doc_topic_distr)
                
                topic_labels_filename = f'{results_dir}/topic_labels_{best_k}.pkl'
                u.save_object(topic_labels_dict, topic_labels_filename)
                logger.info('Saved topic labels in %s', topic_labels_filename)
                
                # Get top 100 words for each topic
                vocab = u.load_object(vocab_filename)
                topic_words_df = get_top_100_words(topic_word_distr, vocab, topic_labels_dict, results_dir)
                
                # Visualize top words
                get_treemap_for_topics(topic_words_df, results_dir)

                # Topics comparison
                results_dir = f'{root_dir}/{decade}'
                jsd_matrix = get_jsd_between_topics(doc_topic_distr, topic_labels_dict, results_dir, best_k)
                similarity_matrix = get_cosine_sim_between_topics(doc_topic_distr, topic_labels_dict, results_dir, best_k)
                
                # Save matrices
                u.save_dense_matrix(jsd_matrix, f'{results_dir}/jsd_matrix_{best_k}.txt')
                u.save_dense_matrix(similarity_matrix, f'{results_dir}/cosine_similarity_matrix_{best_k}.txt')

def get_topic_labels(dtm_filename, vocab_filename, topic_word_distr, doc_topic_distr):
    """ Get topic labels.
    Args:
        None
    Returns:
        None
    """

    vocab = u.load_object(vocab_filename)
    dtm= sp.sparse.load_npz(dtm_filename)
    doc_len = doc_lengths(dtm)

    logger.info('Shapes - topic_word_distr: %s, doc_topic_distr: %s, doc_len: %s, vocab: %s',
            topic_word_distr.shape, doc_topic_distr.shape, doc_len.shape, np.array(vocab).shape)
    
    topic_labels = generate_topic_labels_from_top_words(
        topic_word_distr,
        doc_topic_distr,
        doc_len,
        np.array(vocab),
        lambda_=0.6
    )

    logger.info('Topic labels: %s', topic_labels)
    topic_labels_dict = {i: topic_labels[i] for i in range(len(topic_labels))}
    logger.debug('Topic labels dict: %s', topic_labels_dict)
    
    return topic_labels_dict

def get_cosine_sim_between_topics(doc_topic_distr, topic_labels_dict, results_dir, best_k):
    logger.debug('Document-topic distribution shape: %s', doc_topic_distr.shape)

    # similarity_matrix = cosine_similarity(doc_topic_distr.T) #NOTE: This computes cosine similarity, not distance
    similarity_matrix = 1 - cosine_similarity(doc_topic_distr.T)

    logger.debug('Cosine similarity matrix shape: %s', similarity_matrix.shape)

    plt.figure(figsize=(10, 10))
    
    # Create a clustermap with the row dendrogram
    row_linkage = linkage(similarity_matrix, method='average')

    cmap = sns.light_palette("blue", reverse=True, as_cmap=True)
    sns.set_theme(font_scale=1.4)
    g = sns.clustermap(similarity_matrix,
                        row_linkage=row_linkage,
                        cmap=cmap,
                        vmin=0, 
                        vmax=1,
                        xticklabels=[topic_labels_dict[i] for i in range(len(topic_labels_dict))],
                        yticklabels=[topic_labels_dict[i] for i in range(len(topic_labels_dict))], annot=True, 
                        fmt=".2f", 
                        annot_kws={"size": 16},
                        cbar_kws={'label': 'Cosine Distance', 'ticks': [0.4, 0.6, 0.8]})

    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # Rotate y-tick labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)  # Rotate x-tick labels
    plt.savefig(f'{results_dir}/sim_heatmap_{best_k}.png', bbox_inches='tight')
    
    return similarity_matrix

def get_jsd_between_topics(doc_topic_distr, topic_labels_dict, results_dir, best_k):
    logger.debug('Document-topic distribution shape: %s', doc_topic_distr.shape)

    # Compute pairwise JSD for topic distributions
    # Transpose to get (n_topics, n_documents) for computing distances between topics
    jsd_matrix = distance.pdist(doc_topic_distr.T, metric='jensenshannon') #NOTE: doc-topic distr is transposed to get the distances between topics instead of documents. For the knowledge graph, we would need the distances between documents instead.
    jsd_matrix = distance.squareform(jsd_matrix)

    logger.debug('Jensen-Shannon Divergence matrix shape: %s', jsd_matrix.shape)

    plt.figure(figsize=(10, 10))
    
    # Create a clustermap with the row dendrogram
    row_linkage = linkage(jsd_matrix, method='average')

    cmap = sns.light_palette("blue", reverse=True, as_cmap=True)
    sns.set_theme(font_scale=1.4)
    g = sns.clustermap(jsd_matrix,
                        row_linkage=row_linkage,
                        cmap=cmap,
                        vmin=0, 
                        vmax=1,  # JSD is bounded between 0 and 1
                        xticklabels=[topic_labels_dict[i] for i in range(len(topic_labels_dict))],
                        yticklabels=[topic_labels_dict[i] for i in range(len(topic_labels_dict))], annot=True, 
                        fmt=".2f", 
                        annot_kws={"size": 16},
                        cbar_kws={'label': 'JS Distance', 'ticks': [0.4, 0.6, 0.8]})

    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # Rotate y-tick labels
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)  # Rotate x-tick labels
    plt.savefig(f'{results_dir}/jsd_heatmap_{best_k}.png', bbox_inches='tight')
    
    return jsd_matrix


def get_top_100_words(topic_word_distr, vocab, topic_labels_dict, results_dir):
    """ Get the top 100 words.
    Args:
        None
    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info('Getting 10 top words for each topic')
    
    logger.debug('Topic-word distribution shape: %s', topic_word_distr.shape)

    df = ldamodel_top_topic_words(topic_word_distrib=topic_word_distr,
                                vocab=vocab,
                                top_n=50)
    df.columns = [np.arange(1, len(df.columns) + 1)]
    df.index = [topic_labels_dict[i] for i in topic_labels_dict]
    df.to_csv(f'{results_dir}/topics_top_50_words.csv', index=True)
    return df

def get_treemap_for_topics(topic_words_df, results_dir):
    logger = logging.getLogger(__name__)
    logger.info('Getting treemap for topics')
    df = topic_words_df.copy()  # Make a copy to avoid modifying original
    df = df.T  # Transpose so topics become columns
    df = df.reset_index(drop=True)  # Reset index and drop the old index column
    
    logger.info('Creating treemap for each topic')
    for col in df.columns:
        # Split the column name to get the word and its weight
        word_prob_df = df[col].str.split(' ', expand=True)
        word_prob_df.columns = ['Word', 'Probability']
        # Remove parentheses from probability column
        word_prob_df['Probability'] = word_prob_df['Probability'].str.replace('(', '')
        word_prob_df['Probability'] = word_prob_df['Probability'].str.replace(')', '')
        word_prob_df['Probability'] = word_prob_df['Probability'].astype(float)
        logger.debug('Word probability DF shape: %s', word_prob_df.shape)

        # Extract only the topic label without the number prefix
        clean_topic_label = col.split('_', 1)[1] if '_' in col else col
        
        
        fig = px.treemap(word_prob_df, 
                        path=['Word'], 
                        values='Probability', 
                        color='Probability',
                        color_continuous_scale='RdBu',
                        title=f'Topic "{clean_topic_label}" top 50 words',
                        color_continuous_midpoint=np.average(word_prob_df['Probability']))
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='white', margin=dict(l=0, r=0, t=40, b=0))
        fig.write_image(f'{results_dir}/treemap_topic_{col}.png')
        logger.info('Saved treemap for topic %s', col)
        

