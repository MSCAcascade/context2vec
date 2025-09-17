""" This is the project's access point.
"""
import os
import logging
from src.logger_config import setup_logging
from src.input_output import get_arguments
from src.data_cleaning import get_data2df
from src.tasks.classification.model import get_topics, get_topic_words
from src.tasks.specialization.key_topics import get_key_topics
from src.utils import plot_papers4decade
from src.tasks.specialization.entropy import run_entropy_analysis
from src.tasks.clustering.hac import get_clusters, get_acid_features
from src.tasks.linking.kg import get_percentiles
# Set up logging configuration
setup_logging()

INPUT_FILEPATH = "data/RSC604/rsc_v6_0_4_open_web-export.txt"
DF_FILEPATH = "results/articles_16to18.csv"
def main():
    # Set up logging configuration
    logger = logging.getLogger(__name__)
    
    # Get input arguments
    args = get_arguments()
    logger.info(f'Arguments: {args}')
    
    task = args.task
    input_file_path = INPUT_FILEPATH
    
    # Execute the task
    tasks = ["tm-eval",
            "tm-topics",
            "specialization",
            "clustering",
            "linking",
            "data",
            "eda"]
    if task not in tasks:
        logger.error(f'Task {task} not recognized.')
        return
    if task == "tm-eval":
        logger.info("Getting topics...")
        get_topics()
        logger.info("Topic modeling complete.")
    elif task == "tm-topics":
        logger.info("Getting topic words...")
        #get_topic_words()
        entropy_df = run_entropy_analysis()
        logger.info("Topic words extraction complete.")
    elif task == "specialization":
        get_key_topics()
    elif task == "data":
        logger.info("Data preprocessing...")
        get_data2df(filename=input_file_path)
        logger.info("Data preprocessing complete.")
    elif task == "eda":
        plot_papers4decade(DF_FILEPATH)
    elif task == "clustering":
        #get_clusters()
        get_acid_features()
    elif task == "linking":
        get_percentiles()
        
if __name__ == '__main__':
    main()