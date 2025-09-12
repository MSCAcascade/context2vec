""" This module calculates the entropy values per topic and plots them.
Functions: 
    get_entropy_values(filename): Get entropy values per topic.
    calculate_entropy(topic_probs): Calculate entropy value.
    average_entropy(doc_topic_dist): Calculate average entropy per topic.
    plot_topic_entropy(topic_entropy_values): Plot entropy values per topic.
"""
import os
import src.utils as u
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy
SAVE_PATH = 'results'
MIN_TOPICS = 2
MAX_TOPICS = 30
BEST_K = [6,6,6,6,6,6]
# def get_entropy_values(filename):
#     """Get entropy values per topic.
#     Args:
#         filename: The filename of the dense matrix.
#     Returns:
#         None
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('Starting entropy calculation')

#     doc_topic_dist = u.load_dense_matrix(filename)
#     topic_entropy_values = average_entropy(doc_topic_dist)
#     plot_topic_entropy(topic_entropy_values)
#     logger.info('Completed entropy calculation')

def calculate_entropy(topic_probs):
    """ Calculate entropy value.
    Args:
        topic_probs: A list of probabilities.
    Returns:
        entropy_value: A float value.
    """
    # Ensure probabilities sum up to 1 by normalizing
    topic_probs /= np.sum(topic_probs)
    # Remove zeros to avoid log(0) issue
    topic_probs = topic_probs[topic_probs != 0]
    entropy_value = entropy(topic_probs, base=np.e)
    return entropy_value

def average_entropy(doc_topic_dist):
    """ Calculate average entropy per topic.
    Args:
        doc_topic_dist: A dense matrix.
    Returns:
        entropies: A list of entropy values.
    """
    num_topics = doc_topic_dist.shape[1]
    entropies = []
    for i in range(num_topics):
        topic_probs = doc_topic_dist[:, i]
        entropy_value = calculate_entropy(topic_probs)
        entropies.append(entropy_value)
    return entropies

# def plot_topic_entropy(topic_entropy_values):
#     """ Plot entropy values per topic.
#     Args:
#         topic_entropy_values: A list of entropy values.
#     Returns:
#         None
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('Plotting entropy values per topic')

#     topic_label_dict = {0:'protein',
#                         1:'vaccine',
#                         2:'patient',
#                         3:'cell',
#                         4:'drug',
#                         5:'sample',
#                         6:'health'}

#     topic_entropy_df = pd.DataFrame({'topic': list(topic_label_dict.values()), 'entropy': topic_entropy_values})

#     topic_w_max_entropy = topic_entropy_df.loc[topic_entropy_df['entropy'].idxmax()]
#     logger.info('Topic with highest entropy: %s', topic_w_max_entropy)
#     topic_w_min_entropy = topic_entropy_df.loc[topic_entropy_df['entropy'].idxmin()]
#     logger.info('Topic with lowest entropy: %s', topic_w_min_entropy)

#     x = np.arange(len(topic_entropy_values))
#     y = np.array(topic_entropy_values)

#     plt.scatter(x, y, marker='o', color='black')
#     plt.ylim(6, 8)
#     plt.yticks(np.arange(6, 8.2, 0.2), ['{:.2f}'.format(i) for i in np.arange(6, 8.2, 0.2)])
#     plt.xticks(x, [topic_label_dict[i] for i in x], rotation=45)
#     plt.grid(axis='y', linestyle='--', alpha=0.6)
#     plt.xlabel('Topic')
#     plt.ylabel('Entropy')
#     plt.title('Shannon entropy values per topic')

#     save_path = 'results/entropy'
#     u.create_dir_if_not_exists(save_path)

#     plt.savefig(f'{save_path}/entropy_per_topic.png', bbox_inches='tight')
#     topic_entropy_df.to_csv(f'{save_path}/entropy_per_topic.csv', index=False)


KEY_TOPICS_WPAST = ['wind', 'air', 'water']
                    #'author','year','observation','distance','plant','water','foot','blood']
KEY_TOPICS_WOPAST = ['wind', 'air', 'acid', 'water']
#                     'experiment','point','ball','body','inch','vessel','day','animal','eye','plant','observation','bird','star','gravity','earthquake','day','ray','number','year','thermometer','gold','inscription']

def run_entropy_analysis():

    topic_labels = []
    approach = ['w-past', 'wo-past']
    key_topics = {'w-past': KEY_TOPICS_WPAST, 'wo-past': KEY_TOPICS_WOPAST}
    
    for app in tqdm(approach, desc='Evaluating for 2 approaches: w-past and wo-past'):
        root_dir = f'{SAVE_PATH}/{app}'
        decade_dirs = os.listdir(root_dir) 
        decade_dirs.sort()
        matching_labels = []
        topic_labels = []
        for decade in tqdm(decade_dirs, leave=False):
            # Load data
            doc_topic_distr = u.load_dense_matrix(f'{root_dir}/{decade}/topics_2_30/doc_topic_distr_6.txt')
            topic_labels_dict = u.load_object(f'{root_dir}/{decade}/topics_2_30/topic_labels_6.pkl')
            topic_labels_dict = {k: v.split('_')[1] if '_' in v else v for k, v in topic_labels_dict.items()}
            
            # Check which key topics match labels
            matching_labels = [k for k in key_topics[app] if k in topic_labels_dict.values()]
            if matching_labels:
                # Find indexes of matching labels in topic labels dict
                key_topic_indexes = [list(topic_labels_dict.values()).index(label) for label in matching_labels]
                # For each key topic index, get the doc topic distr
                for index in key_topic_indexes:
                    topic_label = list(topic_labels_dict.keys())[index]
                    topic_labels.append((decade, topic_label, topic_labels_dict[topic_label], doc_topic_distr[:, index]))

        # Save topic labels to file
        topic_labels_path = f'{SAVE_PATH}/topic_labels_{app}.pkl'
        u.save_object(topic_labels, topic_labels_path)
        
        # Get entropy for each topic label
        entropy_results = []
        for decade, topic_label, label_name, topic_probs in tqdm(topic_labels, desc='Calculating entropy for each topic label'):
            entropy_value = calculate_entropy(topic_probs)
            entropy_results.append({
                'decade': decade,
                'topic_label': topic_label,
                'target_label': label_name,
                'shannon_entropy': entropy_value,
                'normalized_entropy': entropy_value / np.log(len(topic_probs)),
                'topic_proportion': np.mean(topic_probs)
            })
        # Create DataFrame from results
        entropy_df = pd.DataFrame(entropy_results)
        entropy_df['year'] = entropy_df['decade'].str.extract(r'(\d{4})').astype(int) // 10 * 10
        entropy_df = entropy_df.sort_values(by=['year', 'target_label', 'topic_label']).reset_index(drop=True)
        # Save entropy DataFrame
        entropy_df_path = f'{SAVE_PATH}/entropy_{app}.csv'
        entropy_df.to_csv(entropy_df_path, index=False)
        logger.info(f'Saved entropy results to {entropy_df_path}')
        # Plot entropy values
        plt.figure(figsize=(9, 6))
        for label in entropy_df['target_label'].unique():
            subset = entropy_df[entropy_df['target_label'] == label]
            plt.plot(subset['year'], subset['shannon_entropy'], marker='o', label=label)
        plt.title(f'Topic entropy ({app})', fontsize=16)
        plt.xlabel('decade', fontsize=14)
        plt.ylabel('entropy', fontsize=14)
        # Set x-ticks to show all decades in your data
        unique_decades = sorted(entropy_df['year'].unique())
        plt.xticks(unique_decades, fontsize=12, rotation=45)
        plt.yticks(fontsize=12)
        plt.grid()
        plt.tight_layout()
                # Place legend outside the plot area
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plot_path = f'{SAVE_PATH}/entropy_plot_{app}.png'
        plt.savefig(plot_path)
        logger.info(f'Saved entropy plot to {plot_path}')
        plt.close()
    return entropy_df