# # L I B R A R I E S

#from Biopython import SeqIO
#import numpy as np
import pandas as pd
import os, glob
#import seaborn as sns
#import matplotlib.pyplot as plt
#from multiprocessing import Pool
from tqdm import tqdm
#from itertools import groupby
#from sklearn.decomposition import LatentDirichletAllocation as LDA
#from sklearn.feature_extraction.text import CountVectorizer
#import math
import json
import re
#from functools import partial
#from collections import defaultdict

# # C R E A T I O N: k_mer_list

folder_path = '/home/may/PostDoct_Tere/COVID_sofi/k_mers'
file_list = os.listdir(folder_path)
file_list.sort(key=lambda f: int(re.sub('\D', '', f))) # Sorts filenames in ascending order 

PATH_files = "/home/may/PostDoct_Tere/COVID_sofi/"
#k_mer_list = []
#for x in tqdm(file_list):
#  k_mer_aux = []
#  filename = folder_path + "/" + x
#  with open(filename, "r") as fp:   # Unpickling
#    k_mer_aux = json.load(fp)
#  k_mer_list.extend(k_mer_aux)
#with open(PATH_files + "complete_kmer_list.txt", "w") as txt_file:   #Serialize
#  json.dump(k_mer_list, txt_file)
#print(len(k_mer_list)) #89384

# # C R E A T I O N: Data_Frame w/ k_mer_list 
#print('Loading DataFrame...')
#Data_Frame = pd.read_csv("/home/may/PostDoct_Tere/COVID_sofi/COVID_complete.csv")
print('Loading k-mer list...')
kmer_path = "/home/may/PostDoct_Tere/COVID_sofi/complete_kmer_list.txt"
with open(kmer_path, "r") as fp:   # Unpickling
    k_mer_list = json.load(fp)
print('Creating DataFrame...')
dic = {'k-mers': k_mer_list}
df_kmers = pd.DataFrame(dic)
#df_concat = pd.concat([Data_Frame,df_kmers],axis=1)
#df_concat.to_csv(PATH_files + 'COVID_kmers_complete.csv')