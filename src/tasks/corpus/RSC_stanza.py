import os
import torch
import multiprocessing as mp
from multiprocessing import Process
from tqdm import tqdm

def worker(file_list, gpu_id, input_folder, output_folder):
    """
    Process a list of files with stanza on a specific GPU.

    This function processes a list of files in parallel using multiple GPUs. It is
    designed to be used with multiprocessing.Pool.
    For each file, it tokenizes, POS-tags, lemmatizes, and dependency-parses the text. Crucially, it splits the text into sentences, then outputs each sentence in a <s>…</s> block in the output file.

    Parameters
    ----------
    file_list : list of str
        List of file names to process.
    gpu_id : int
        ID of the GPU to use.
    input_folder : str
        Folder containing the input files.
    output_folder : str
        Folder to save the output files.

    Returns
    -------
    None
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse', use_gpu=True)

    error_files = []  # save the list of erronous files

    for file_name in tqdm(file_list, desc=f"GPU {gpu_id}"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

            doc = nlp(text)

            with open(output_path, 'w', encoding='utf-8') as out_f:
                for i, sentence in enumerate(doc.sentences):
                    out_f.write(f'<s id={file_name}_{i}>\n')
                    for word in sentence.words:
                        out_f.write(f'{word.text}\t{word.upos}\t{word.lemma}\t{word.id}\t{word.head}\t{word.deprel}\n')
                    out_f.write('</s>\n')

        except Exception as e:
            print(f"Error processing {file_name} on GPU {gpu_id}: {e}")
            error_files.append(file_name)

    # Save the list of erronous files
    if error_files:
        error_log_path = os.path.join(output_folder, f'error_files_gpu{gpu_id}.txt')
        with open(error_log_path, 'w', encoding='utf-8') as f:
            for ef in error_files:
                f.write(f"{ef}\n")


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

if __name__ == '__main__':
    mp.set_start_method('spawn') # So that it does not copy the stanza from 1 GPU to another

    input_folder = '/home/volt/bach/Corpora/RSC/Royal_Society_Corpus_open_v6.0_texts_txt'
    output_folder = '/home/volt/bach/Corpora/RSC/Royal_Society_Corpus_open_v6.0_texts_txt_stanza'
    os.makedirs(output_folder, exist_ok=True)

    file_list = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    n = 4  # số GPU
    file_lists = split_list(file_list, n)

    processes = []
    for gpu_id in range(n):
        p = Process(target=worker, args=(file_lists[gpu_id], gpu_id, input_folder, output_folder))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    print("Done")
