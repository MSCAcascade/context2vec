# Reporting
import logging
logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
# File manipulation
import os
import pickle
# Tensor manipulation
import pandas as pd
import numpy as np
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

def plot_papers4decade(filename):
    # NOTE: Uncomment and change parameter if input = filepath
    df = pd.read_csv(filename)
    
    decades = ['1750', '1760', '1770', '1780', '1790', '1800']
    # Convert 'decade' column to string if it is not already
    if df['decade'].dtype != 'object':
        df['decade'] = df['decade'].astype(str)
    # Drop rowns which decade is not in decades
    df = df[df['decade'].isin(decades)]
    
    # Count number of papers per decade
    df_count = df['decade'].value_counts().reset_index()
    df_count.columns = ['decade', 'count']
    # Sort by decade
    df_count = df_count.sort_values(by='decade')
    # Plotting
    plt.figure(figsize=(7, 6))
    sns.lineplot(x='decade', y='count', data=df_count, marker='o', color='blue')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    total_papers = df_count['count'].sum() # Create legend with N = total papers
    plt.legend([f'N: {total_papers}'], fontsize=12)
    plt.title('Publications per decade', fontsize=16)
    plt.xlabel('decade', fontsize=14)
    plt.ylabel('counts', fontsize=14)
    plt.tight_layout()
    
    # Save plot
    results_folder = 'results'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    plot_path = os.path.join(results_folder, 'publications_per_decade_1750to1800.png')
    plt.savefig(plot_path)
    logger.debug(f'Plot saved to {plot_path}')

def save_object(object, save_path):
    """ Save an object.
    Args:
        object (object): The object to save.
    Returns:
        None
    """
    with open(save_path, 'wb') as f:
        pickle.dump(object, f)

def load_object(load_path):
    """ Load an object.
    Args:
        load_path (str): The path to the object.
    Returns:
        object: The loaded object.
    """
    with open(load_path, 'rb') as f:
        return pickle.load(f)
    
def save_dense_matrix(matrix, save_path):
    """ Save a dense matrix.
    Args:
        matrix (np.array): The matrix to save.
        save_path (str): The path to save the matrix.
    Returns:
        None
    """
    np.savetxt(save_path, matrix, delimiter='\t', fmt='%1.5f')

def load_dense_matrix(filename):
    """ Load a dense matrix.
    Args:
        filename (str): The filename.
    Returns:
        np.array: The matrix.
    """
    matrix = np.loadtxt(filename, delimiter='\t')
    return matrix