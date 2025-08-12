# Internal 
import src.utils as u
# Reporting
import random
random.seed(42)
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
import warnings
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
# File manipulation
import os
# Tensor manipulation
import pandas as pd
import numpy as np
import scipy as sp
# Topic modeling
from tmtoolkit.utils import enable_logging, disable_logging
from tmtoolkit.corpus import (load_corpus_from_picklefile, dtm)
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.bow.bow_stats import doc_lengths
from tmtoolkit.topicmod.model_stats import generate_topic_labels_from_top_words
from tmtoolkit.topicmod.model_io import ldamodel_top_topic_words
# Visualization
from tmtoolkit.topicmod.visualize import generate_wordclouds_for_topic_words, plot_eval_results
import seaborn as sns
import matplotlib.pyplot as plt
LIMIT = None
N_ITER = 1000
RANDOM_STATE = 20191122
ETA = 0.1
MIN_TOPICS = 2
MAX_TOPICS = 30
STEP_TOPICS = 2
def models_evaluation(filename, results_dir):
    dtm = sp.sparse.load_npz(filename)
    
    #NOTE: Change limit if not in development
    if LIMIT is not None:
        dtm = dtm[:LIMIT, :]

    disable_logging()
    logger = logging.getLogger('lda')
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    warnings.filterwarnings('ignore')
    
    const_params = {
        'n_iter': N_ITER,
        'random_state': RANDOM_STATE,
        'eta': ETA
    }

    var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(MIN_TOPICS, MAX_TOPICS, STEP_TOPICS)]

    metrics = ['loglikelihood', 'coherence_mimno_2011']

    # Evaluation
    eval_results = evaluate_topic_models(dtm,
                                        varying_parameters=var_params,
                                        constant_parameters=const_params,
                                        metric=metrics,
                                        return_models=True)
    
    # Plot
    min_topics = str(MIN_TOPICS)
    max_topics = str(MAX_TOPICS)

    save_path = f'{results_dir}/topics_' + min_topics + '_' + max_topics
    os.makedirs(save_path, exist_ok=True)
    u.save_object(eval_results, f'{save_path}/eval_results.pkl')

    eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
    u.save_object(eval_results, f'{save_path}/eval_results_by_topics.pkl')

    xaxislabel = 'topics'
    title = ['Perplexity\n(minimize)', 'Topic coherence\n(maximize)']
    yaxislabel = ['score', 'score']

    for index in range(0, len(metrics)):
        fig, subfig, axes = plot_eval_results(eval_results_by_topics,
                                            metric=metrics[index],
                                            xaxislabel=xaxislabel, 
                                            yaxislabel=yaxislabel[index], 
                                            show_metric_direction=False,
                                            figsize=(12, 6),
                                            title=title[index])

        for ax in axes:
            ax.set_title("")
            ax.grid(True)
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            ax.set_xticks(np.arange(MIN_TOPICS, MAX_TOPICS, STEP_TOPICS))
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.ticklabel_format(style='sci')
            ax.yaxis.get_offset_text().set_fontsize(16)

        plt.savefig(f'{save_path}/plot_eval_results_{metrics[index]}.png')