""" This module contains functions to parse input arguments and write output files. 
"""
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Experiment configuration')
    parser.add_argument('task', type=str, help='The task to perform')
    args = parser.parse_args()
    return args