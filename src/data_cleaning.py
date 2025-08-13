# Internal
import src.utils as u
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
# String manipulation
import re
import string
# Topic modeling
from tmtoolkit.corpus import (Corpus, save_corpus_to_picklefile, load_corpus_from_picklefile, print_summary, lemmatize, filter_for_pos, to_lowercase, remove_punctuation, filter_clean_tokens, remove_common_tokens, remove_uncommon_tokens, tokens_table, dtm)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LIMIT = 20
TEXTS_INPERIOD = "./data/rscv604_textId_year"
SAVE_PATH = "results"
CPUS = 4

def extract_articles(text):
    """ Extract articles from a text file.
    Args:
    text (str): The text content of the file.
    Output:
    list: A list of tuples (text_id, article_text).
    """
    articles = []
    current_text_id = None
    current_text = []
    inside_text = False

    for line in text.split('\n'):
        if line.startswith('<text id='):
            if current_text_id is not None:
                articles.append((current_text_id, ' '.join(current_text)))
            current_text_id = line.split('"')[1].strip()
            current_text = []
            inside_text = True
        elif line.startswith('</text>'):
            if current_text_id is not None:
                articles.append((current_text_id, ' '.join(current_text)))
            current_text_id = None
            current_text = []
            inside_text = False
        elif inside_text:
            current_text.append(line)

    if LIMIT is not None:
        random.shuffle(articles)
        articles = articles[:LIMIT]
        logger.debug(f'Number of articles after shuffling and limiting: {len(articles)}')

    return articles

def check_targets_presence(text):
    pattern = r'\w*oxyge\w*|\w*phlogist\w*|\w*acid\w*|water\w*|gas\w*|\w*hydro\w*|substance\w*|solution\w*|\w*oxid\w*|compound\w*|muriatic\w*|\w*combust\w*|\w*flam\w*|electric\w*|lumin\w*|ether|caloric|air|heat|fire|energy|\w*radical\w*|potential\w*|metal\w*'
    text = str(text)  # Ensure text is a string
    matches = re.findall(pattern, text)
    if matches:
        score = 1
    else:
        score = 0
    return score

def filter_uppercase_tokens(text): #NOTE: Not used, maybe useful for future work
    # Regex for all-caps names/groups (e.g., J. EVELYN, JOHN DOE)
    pattern = r'\b(?:[A-Z]\.\s*)?(?:[A-Z]+\.?\s*){2,}'
    text = str(text) 
    try:
        matches = re.findall(pattern, text)
        cleaned_matches = [m.strip().rstrip('.') for m in matches]
    except Exception as e:
        logger.error(f"Error finding uppercase tokens: {e}")
        cleaned_matches = []
    return cleaned_matches

def clean_data_round1(text):
    text = text.lower()
    text = re.sub(r'\b\w*([^\w\s]|\d)\w*\b', '', str(text))  # Removes words with special characters or digits
    text = re.sub('[‘’“”…]', '', str(text)) # Removes special characters
    #NOTE: text = re.sub('-', '', str(text)) # Compound words become one word
    text = re.sub('[%s]' % re.escape(string.punctuation), '', str(text)) # Removes punctuation
    text = re.sub(r'\d+', '', text)  # Removes all digits
    text = re.sub(r'\b\w{1}\b', '', str(text))  # Removes single-character words
    text = re.sub(r'\s+', ' ', str(text)).strip() # Removes extra spaces and leading/trailing spaces
    return text

def clean_data_round2(filename):
    """ Corpus formation. This function creates a corpus from a tabular file."""
    corpus = Corpus.from_tabular(filename, language='en', id_column='text_id', text_column='article_text', max_workers=CPUS)
    lemmatize(corpus)
    filter_for_pos(corpus,search_pos=['N','ADJ'],simplify_pos=True, tagset='penn')
    #remove_punctuation(corpus_norm) #NOTE: Uncomment to concat hypens, e.g., "sulphuric-acid" -> "sulphuricacid"
    # Get DTM
    dtm_sparse, doc_labels, vocab = dtm(corpus, return_doc_labels=True, return_vocab=True)
    logger.debug('Clean corpus shape: %s', dtm_sparse.shape)

    return corpus, dtm_sparse, vocab, doc_labels

def get_data2df(filename):
    """ This function reads a list of texts from a file and returns a DataFrame (textId x clean text).
    """
    # Data loading
    logger.info(f"1-Loading data from {filename}...")
    with open(filename, 'r') as file:
        text = file.read()
    articles = extract_articles(text)
    logger.info(f'Extracted {len(articles)} articles.')
    
    # Sampling by range (1750-1800)
    df = pd.DataFrame(articles, columns=['text_id', 'article_text'])
    logger.info(f'2-Sampling decades based on {TEXTS_INPERIOD}...')
    df_inperiod = pd.read_csv(TEXTS_INPERIOD, sep='\t', header=None, names=['text_id', 'year'])
    df['text_id'] = df['text_id'].astype(str)
    df_inperiod['text_id'] = df_inperiod['text_id'].astype(str)
    merged_df = pd.merge(df, df_inperiod, on='text_id')
    logger.debug(f'Merged DataFrame shape: {merged_df.shape}')
    
    # Sampling by oxy-terms
    logger.info("3-Filtering articles with oxy-terms...")
    scores = []
    for k in range(0, len(merged_df.article_text)):
        scores.append(check_targets_presence(merged_df.article_text[k]))
    total = np.sum(scores)
    logger.debug(f'Total doc/oxy-terms matches: {total}')

    not_oxy_terms = [i for i, score in enumerate(scores) if score == 0]
    merged_df = merged_df.drop(merged_df.index[not_oxy_terms])
    logger.info(f'Total articles with oxy-terms: {merged_df.shape}')
    
    logger.debug("Visualizing publications by decade...")
    u.plot_papers4decade(merged_df)
    
    # NOTE: Uncomment to filter all uppercase words, commonly author names
    # logger.info("Filtering uppercase tokens..")
    # uppercase_tokens = []
    # for k in range(0, len(merged_df.article_text)):
    #     uppercase_tokens.append(filter_uppercase_tokens(merged_df.article_text[k]))

    # ids = merged_df['text_id'].tolist()
    # dict_uppercase = {'text_id': ids, 'uppercase_tokens': uppercase_tokens}
    # df_uppercase = pd.DataFrame(dict_uppercase)
    # del ids, dict_uppercase
    # logger.debug(f'Sample of uppercase tokens: {df_uppercase.head(2).to_dict()}')
    
    # Data cleaning 1: lowercase, remove special characters, digits, punctuation, single-character words, stopwords
    logger.info("4-Cleaning data round 1...")
    merged_df['article_text'] = merged_df['article_text'].apply(lambda x: clean_data_round1(x))
    logger.debug(f'Sample of cleaned text: {merged_df["article_text"].head(1).tolist()}')
    
    logger.info("Removing stopwords...")
    stop_scikit = ENGLISH_STOP_WORDS
    merged_df['article_text'] = merged_df['article_text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop_scikit)]))
    logger.debug(f'Sample of text after stopword removal: {merged_df["article_text"].head(1).tolist()}')
    
    # Creating DF subsets for each decade (cumulative, non-cumulative)
    logger.info("5-Cleaning data round 2...")
    
    intervals = np.arange(1750, 1801, 5)
    decades = [f"{start}-{start+4}" for start in intervals[:-1]]
    logger.debug(f'Decades: {decades}')
    merged_df['decade'] = pd.cut(merged_df['year'], bins=intervals, labels=decades, right=False)
    logger.debug(f'Merged DataFrame with decades: {merged_df.head(2).to_dict()}')
    merged_df = merged_df.drop(columns=['year'])
    logger.debug(f'Merged DataFrame shape after decade assignment: {merged_df.shape}')
    logger.info("Creating subsets for each decade...")
    
    approach = ['w-past', 'wo-past']
    for decade in tqdm(decades, desc="Processing decades"):
        for app in tqdm(approach, leave=False):
            results_folder = f'{SAVE_PATH}/{app}/{decade}'
            os.makedirs(results_folder, exist_ok=True)
            if app == 'wo-past':
                df_to_save = merged_df[merged_df['decade'] == decade].copy()  # Only current decade
            else:  # w-past
                df_to_save = merged_df[merged_df['decade'] <= decade].copy()  # Current + all previous decades
            
            csv_path = f'{results_folder}/df_{decade}.csv'
            df_to_save.to_csv(csv_path, index=False)
            
            # Data cleaning 2: lemmatization, POS filtering, DTM/decade
            corpus, dtm, vocab, doc_labels = clean_data_round2(csv_path)
            u.save_object(vocab, f'{results_folder}/vocab.pkl')
            u.save_object(doc_labels, f'{results_folder}/doc_labels.pkl')
            sp.sparse.save_npz(f'{results_folder}/dtm_sparse', dtm)
            save_corpus_to_picklefile(corpus, f'{results_folder}/clean_corpus.pkl')
            logger.debug(f'Saved DTM, corpus, doc_labels, and vocab for {decade} in {results_folder}')