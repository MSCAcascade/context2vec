# # # L I B R A R I E S
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import tmtoolkit
from tmtoolkit.topicmod import tm_lda
from tmtoolkit.topicmod.tm_lda import evaluate_topic_models
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results
from scipy.stats import entropy
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.spatial import distance
from scipy.sparse import vstack
from scipy.sparse import csr_matrix
from gensim.models import ldamodel
from gensim.matutils import jensen_shannon
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import os, glob
import re
import json

# # C R E A T I O N: k_mer_list
"""
folder_path = '/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/k_mers'
file_list = os.listdir(folder_path)
file_list.sort(key=lambda f: int(re.sub('\D', '', f))) # Sorts filenames in ascending order 

PATH_files = "/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/"
k_mer_list = []
for x in tqdm(file_list):
  k_mer_aux = []
  filename = folder_path + "/" + x
  with open(filename, "r") as fp:   # Unpickling
    k_mer_aux = json.load(fp)
  k_mer_list.extend(k_mer_aux)
with open(PATH_files + "complete_kmer_list.txt", "w") as txt_file:   #Serialize
  json.dump(k_mer_list, txt_file)
print(len(k_mer_list)) # 9472

# # C R E A T I O N: Data_Frame w/ k_mer_list 
PATH_files = "/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/"
#print('Loading DataFrame...')
Data_Frame = pd.read_csv("/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/COVID_LATAM_complete.csv")
print('Loading k-mer list...')
kmer_path = "/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/complete_kmer_list.txt"
with open(kmer_path, "r") as fp:   # Unpickling
    k_mer_list = json.load(fp)
print('Creating DataFrame...')
dic = {'k-mers': k_mer_list}
df_kmers = pd.DataFrame(dic)
df_concat = pd.concat([Data_Frame,df_kmers],axis=1)
print(df_concat.iloc[9471]) #tiene NaN en id, location, etc., por que?
df_concat.to_csv(PATH_files + 'COVID_kmers_complete.csv')



# # C O U N T  V E C T O R I Z E R
cv = CountVectorizer()
#col_list = ['k-mers']
Data_Frame = pd.read_csv("/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/COVID_LATAM_kmers_complete.csv")
print(Data_Frame)


#---- Get DTM
path_cv = '/home/daniel/Documents/LDA_Sophie/CountVectorizer/'
path_headers = '/home/daniel/Documents/LDA_Sophie/Headers_CountVectorizer/'
cv = CountVectorizer()
col_list = ['k-mers']
counter = 0
print('Loading CountVectorizer...')
for chunk in pd.read_csv('/home/daniel/Documents/LDA_Sophie/Metadata/COVID_kmers_complete.csv',\
    usecols=col_list, chunksize=1000):
    data_cv = cv.fit_transform(chunk['k-mers'])
    counter = counter + 1
    headers = []
    headers = cv.get_feature_names()
    scipy.sparse.save_npz(path_cv + str(counter) + '_cv_kmers.npz', data_cv)
    with open(path_headers + str(counter) + '_headers.txt', 'w') as txt_file:
      json.dump(headers, txt_file)
    print(counter)
print('CountVectorizer complete')

# ---- Concatenate DTMs

## Read headers
# Directories
path_headers = '/home/daniel/Documents/LDA_Sophie/Headers_CountVectorizer'
headers_list = os.listdir(path_headers)
headers_list.sort(key=lambda f: int(re.sub('\D', '', f))) # Sorts filenames in ascending order
# Get all filenames
headers_filenames = []
for x in tqdm(headers_list):
 headers_filenames.append(path_headers + "/" + x)

## Read data 
# Directories
path_cv = '/home/daniel/Documents/LDA_Sophie/CountVectorizer'
data_list = os.listdir(path_cv)
data_list.sort(key=lambda f: int(re.sub('\D', '', f))) # Sorts filenames in ascending order
# Get all filenames
data_filenames = []
for x in tqdm(data_list):
 data_filenames.append(path_cv + "/" + x)

## numpy array to dataframe

Temporal_DF = []
#for i in tqdm(range(0,len(data_filenames))):
for i in tqdm(range(80,90)): #0-20,20-40,40-60,60-80,80-90
  sparse_matrix = scipy.sparse.load_npz(data_filenames[i]) # Reads data_filename
  compressed_matrix = csr_matrix(sparse_matrix)
  dense_matrix = compressed_matrix.todense()
  with open(headers_filenames[i], 'r') as txt_file:
    header = json.load(txt_file)
    #print(sparse_matrix.shape) # (1000,34133)
    #print(len(header)) # (34133)
  DF = pd.DataFrame(data=dense_matrix, columns=header)
  Temporal_DF.append(DF)

LDA_DF = pd.concat(Temporal_DF, ignore_index = True)
LDA_DF.fillna(0, inplace=True) #downcast='infer')
path_df = "/home/daniel/Documents/LDA_Sophie/DF_CountVectorizer/"
LDA_DF.to_csv(path_df + "DF_5.csv")#,dtype=int32)
# save csv <- dtype=int32
print(LDA_DF)

# ---- Read and concatenate each batch
Temporal_DF = []
# Read DF
print("Loading first DF...")
DF_1 = pd.read_csv("/home/daniel/Documents/LDA_Sophie/DF_CountVectorizer/DF_1.csv") # load df 1
print("Loading second DF...")
DF_2 = pd.read_csv("/home/daniel/Documents/LDA_Sophie/DF_CountVectorizer/DF_2.csv") # load df 2
print("DF load complete")
LDA_DF = pd.concat([DF_1, DF_2], ignore_index = True)
LDA_DF.fillna(0, inplace=True)
print(DF_1.shape)
print(DF_2.shape)
print(LDA_DF.shape)
print(LDA_DF)

# # L D A  E V A L U A T I O N
print("1) Loading dataframe")
cv = CountVectorizer()
col_list = ['k-mers']
Data_Frame = pd.read_csv("/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/COVID_LATAM_kmers_complete.csv",usecols=col_list)
print(Data_Frame)
# Get DTM
print('2) Creating DTM...')
data_cv = cv.fit_transform(Data_Frame['k-mers'])
print('DTM complete')
# Evaluation's parameters definition
print('3) Define evaluation model parameters...')
var_params = [{'n_topics': k, 'alpha': 1/k} for k in range(18, 20, 1)] # 2-4, 4-6, 6-8, 8-10, 10-12, 12-14, 14-16, 16-18, 18-20
const_params = {
    'n_iter': 1000,
    'eta': 0.1,       # "eta" akddubuntua "beta"
    'random_state': 20191122  # to make results reproducible
}
print('Definition complete')
# Evaluation results
print('4) Running LDA evaluation...')
eval_results = evaluate_topic_models(data_cv,
                                     varying_parameters=var_params,
                                     constant_parameters=const_params,
                                     return_models=True,
                                     metric=['arun_2010','cao_juan_2009',
                                             'griffiths_2004','loglikelihood'])
print('LDA evaluation complete')
eval_results_by_topics = results_by_parameter(eval_results, 'n_topics')
plot_eval_results(eval_results_by_topics)
#plt.show()

# Save results into dataframe
topic_number = []
griffiths_values = []
loglikelihood_values = []
arun_values = []
caojuan_values = []
eval_results_path = "/home/daniel/Documents/LDA_Sophie/LDA_Evaluation/"

print('Saving results into dataframe...')
for index in range(len(eval_results_by_topics)):
  topic_number.append(eval_results_by_topics[index][0]) # topics
  griffiths_values.append(eval_results_by_topics[index][1]['griffiths_2004']) 
  loglikelihood_values.append(eval_results_by_topics[index][1]['loglikelihood']) 
  arun_values.append(eval_results_by_topics[index][1]['arun_2010']) 
  caojuan_values.append(eval_results_by_topics[index][1]['cao_juan_2009']) 

eva_dict = {'topic_number':topic_number,'griffiths_2004':griffiths_values,\
            'loglikelihood':loglikelihood_values,'arun_2010':arun_values,\
            'cao_juan_2009':caojuan_values}
eva_df = pd.DataFrame.from_dict(eva_dict)

eva_df.to_csv(eval_results_path + 'LATAM_Evaluation_results_18to20.csv')
print('Results saved')

# Conatenate LDA Evaluation csv files into Dataframe
path_cv = '/home/daniel/Documents/LDA_Sophie/LDA_Evaluation'
data_list = os.listdir(path_cv)
data_list.sort(key=lambda f: int(re.sub('\D', '', f))) # Sorts filenames in ascending order
data_filenames = []
for x in tqdm(data_list):
  data_filenames.append(path_cv + "/" + x) # Get all filenames

Temporal_DF = []
col_list = ['topic_number','griffiths_2004','loglikelihood','arun_2010','cao_juan_2009']
for i in tqdm(range(0,len(data_filenames))): 
  Data_Frame = pd.read_csv(data_filenames[i],usecols=col_list)
  Temporal_DF.append(Data_Frame)

LDA_Evaluation_DF = pd.concat(Temporal_DF, ignore_index = True)

# Plot
fig, axs = plt.subplots(4)
axs[0].plot(LDA_Evaluation_DF['topic_number'],LDA_Evaluation_DF['griffiths_2004'])
axs[0].set_title('griffiths_2004')
axs[1].plot(LDA_Evaluation_DF['topic_number'],LDA_Evaluation_DF['loglikelihood'])
axs[1].set_title('loglikelihood')
axs[2].plot(LDA_Evaluation_DF['topic_number'],LDA_Evaluation_DF['arun_2010'])
axs[2].set_title('arun_2010')
axs[3].plot(LDA_Evaluation_DF['topic_number'],LDA_Evaluation_DF['cao_juan_2009'])
axs[3].set_title('cao_juan_2009')
plt.show()

# #  L D A  M O D E L

def print_topics(model, vectorizer, n_top_words): # Function to print the top words in each topic
    words = vectorizer.get_feature_names() # Get the words from the Corpus
    Topics_lexica = list()
    Word_topic_probability = list()
    for topic_idx, topic in enumerate(model.components_):
        ### Uncomment if you want to display the top words by each topic
        #print('\nTopic #%d:' % topic_idx)
        #print(' '.join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        Topic_word = [words[i] for i in topic.argsort()[::-1]]
        Probability_word = np.sort(topic)[::-1] /np.sum(topic)
        Word_topic_probability.append(Probability_word) 
        Topics_lexica.append(Topic_word)
    return Topics_lexica, Word_topic_probability

# Get DTM
print('1) loading DF')
cv = CountVectorizer()
col_list = ['k-mers']
Data_Frame = pd.read_csv("/home/daniel/Documents/LDA_Sophie/Metadata_07-03-2021/COVID_LATAM_kmers_complete.csv",usecols=col_list)

data_cv = cv.fit_transform(Data_Frame['k-mers'])
print('2) running lda model')

# LDA
n_components = 6
n_top_words = 30
alpha = None
eta = None
columns = cv.get_feature_names()
lda = LatentDirichletAllocation(n_components=n_components, max_iter=80,
                                learning_method='batch',
                                learning_offset=50,
                                random_state=0,
                                doc_topic_prior=alpha,
                                topic_word_prior=eta)
lda.fit(data_cv)
Topic_Probability = lda.transform(data_cv) # numpe.ndarray
Topics_Lexica, Words_Probability = print_topics(lda,cv,n_top_words)
print('lda model complete')

# Results
print('3) saving results')
PATH_topics = "/home/daniel/Documents/LDA_Sophie/LDA_model/"
np.savetxt(PATH_topics + 'Topic_Probability.txt', Topic_Probability, fmt = '%s', delimiter = '\t')
for k in range (0, len(Topics_Lexica)):
    Topic_Information = list(zip(Topics_Lexica[k], Words_Probability[k])) # Concatenate words and its topic probability 
    np.savetxt(PATH_topics + 'Topic_' + str(k) + '_words.txt', Topic_Information, fmt = '%s', delimiter = '\t')
print('results saved')
"""
# Hierarchical clustering
results_file = "/home/daniel/Documents/LDA_Sophie/LDA_model/Topic_Probability.txt"
Topic_Probability = np.loadtxt(results_file, dtype=float, delimiter = '\t')
print(Topic_Probability.shape)
hc = AgglomerativeClustering(n_clusters=10,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(Topic_Probability)
print(y_hc)

plt.scatter(Topic_Probability[y_hc==0, 0], Topic_Probability[y_hc==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(Topic_Probability[y_hc==1, 0], Topic_Probability[y_hc==1, 1], s=100, c='blue', label ='Cluster 2')
plt.scatter(Topic_Probability[y_hc==2, 0], Topic_Probability[y_hc==2, 1], s=100, c='green', label ='Cluster 3')
plt.scatter(Topic_Probability[y_hc==3, 0], Topic_Probability[y_hc==3, 1], s=100, c='cyan', label ='Cluster 4')
plt.scatter(Topic_Probability[y_hc==4, 0], Topic_Probability[y_hc==4, 1], s=100, c='magenta', label ='Cluster 5')
plt.scatter(Topic_Probability[y_hc==5, 0], Topic_Probability[y_hc==5, 1], s=100, c='yellow', label ='Cluster 6')
plt.title('Clusters of Genomic Sequences (Hierarchical Clustering Model)')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100')
plt.show()
