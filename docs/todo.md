# Methodology

Tasks: 
1. Classification of scholarly documents
2. Identification of contextual factors driving revolutionary change

# Python environment
- [X] Define dependencies in requirements.txt: Python version 3.11 for data(don't install tmtoolkit packages for extra metrics), 3.10 for the rest
    - [x] Create env with conda: 
        - `conda create --name myenvironment python=3.10`
        - `pip install -r requirements.txt`
    - [x] Complete tmtoolkit setup: `python -m tmtoolkit setup en`
- [x] Clone KnowFormer
- [ ] #TODO: Application replicability
    - [ ] add data/results to zenodo, code to github
    - [ ] add dockerfile to github, describe setup with bash script in README
    - [ ] create config file w/inputs filenames

# Pre-processing
Input: corpus, metadata (authors, decade)
Output: TF-IDF, citation graph?

- [x] Dataset definition: RSC_V6_0_4_OPEN_WEB
    - Server: corpora.clarin-d.uni-saarland.de
        - RSC_V6_0_1_OPEN_WEB
        - RSC_V6_0_4_OPEN_WEB --with topics (30)
- [x] Sampling
    - [x] Define target decades: 1600-1800
    - [x] Filter documents within period: 17519(total) -> 5348
        - CQP: 1600-1800
            - `<text>[]::match.text_decade="16[0-9]{2}|17[0-9]{2}|1800"`
            - `tabulate Last match text_id, match (feature) >"(filename)"`
    - [x] Filter documents with oxy-terms hits
        - [x] Define keywords pattern: 
            `r'\w*oxyge\w*|\w*phlogist\w*|\w*acid\w*|water\w*|gas\w*|\w*hydro\w*|substance\w*|solution\w*|\w*oxid\w*|compound\w*|muriatic\w*|\w*combust\w*|\w*flam\w*|electric\w*|lumin\w*|ether|caloric|air|heat|fire|energy|\w*radical\w*|potential\w*|metal\w*'`
            - Based on Stefania's study using KLD: https://aclanthology.org/2021.latechclfl-1.14
                - Keywords: ``[word ="oxyge.*|phlogiston|dephlogisticated|acid|water|gas|hydrogen.*|substance|solution|oxide|compound|muriatic"]``
                - Names: Priestley-> Pearson, Pearson->Chevenix, Davy, Henry (see Fig. 2)
            - Based on ChemRevo epistemic objects article: https://doi.org/10.1007/s10670-011-9340-9
                - Keywords: **oxygen**, **phlogiston**, **caloric**, **acidity**, **dephlogisticated** **air**, **fire** air, **combustion**, **energy**, **potential**, **muriatic**, **electricity**, **radical**, **hydrogen**, **gas**, **luminiferous** **ether**, **metal**
                - Names: Lavoisier, Priestley, Wilhelm Scheele
    - [x] Visualizations
        - [x] Get papers distribution by decade
- [x] Quality of vocabulary
    - [x] Pre-cleaning
        - [ ] *Filter all uppercase words -> citations* --deprecated: future work
            - idea: send words to tuple (uppercase words, textId), then apply normal capitalization
        - [x] Remove special characters (outside text), punctuation, numbers (out/in text), words w/ len<2
        - [x] Lowercase
        - [x] Remove stopwords based on standard scikit list
        - [ ] *Homogenize oxygen terms inconsistencies* --deprecated: filtering by oxy-terms solves this
    - [x] Cleaning
        - [ ] *Convert all uppercase words -> normal capitalization* --deprecated: future work ~NER4citations
        - [x] Handle oxygen terms inconsistencies (hydrogene -> hydrogen) --handled above
          - #NOTE: Stefania: Yes, for TM handle spelling inconsistencies
        - [x] Handle hyphenated terms: unchanged
          - #NOTE: Stefania: Leave hyphenated terms to not change the words distribution
        - [x] Use POS tags to filter NN, ADJ (Penn based, see: https://tmtoolkit.readthedocs.io/en/latest/api.html#tmtoolkit.corpus.filter_for_pos)
          - #NOTE: Stefania: consider ADJ, first instance of oxygen is "oxygenous"
          - #NOTE: Bach: considers the dependents or the words the target is dependent on
        - [x] Processing for TM
            - [x] Define strategies: w-past, wo-past
            - [x] Create DTM, corpus, vocab, doc_labels per decade
        - [ ] *Processing for citation graph* --deprecated: future work
            - [ ] Find authors names in text
                - [ ] POS tags -> filter PN #REVIEW: Fine-tune PLM instead of using Stanza?
                - [ ] Homogenize authors names in metadata and references
            - [ ] Filter documents with key names
                - [ ] Lavoisier --see RSC entry [example](http://dx.doi.org/10.1098/rstl.1782.0017)
                    - CQP: `[word = "(?i)\bLavoisier\b"]`
                    - #NOTE: Referenced authors were written in uppercase
    - [ ] #TODO: Visualizations
        - [ ] Plot oxy-terms distribution by decade

## Observations

1. Vocab after preprocessing

```
2025-07-21 12:07:46 | DEBUG | Random words from vocabulary: ['opinion', 'FIGA', 'Animals', 'sober', 'Sub', 'Result', 'Phil', 'Histor', 'sirlgle', 'Electricity', 'probability', 'smoothness', 'induction', 'Deansgate', 'machine', 'cumber', 'Author', 'Artist', 'Ditches', 'Patient', 'Projection', 'gentleman', 'mervs', 'Appearance', 'jealous', 'NB', 'saddleworth', 'paraphrase', 'removed', 'indifferent', 'crimson', 'Perception', 'driest', 'lye', 'Sv', 'ultimatum', '2afi', 'submission', 'tube', 'Jellow', 'refracting', 'curious', 'arsimalsX', 'Surfaces', 'Intestines', 'PH', 'sunlmer', 'applekernel', 'Egyptian', 'Dissections', 'central', 'Dysentery', 'bladder', 'xuale', 'atay', 'military', 'Sine', 'tv', 'Body', 'shark', 'ell', 'igitur', 'Gentleman', 'ccidental', 'Creator', 'intention', 'Unicorns', 'vzjitll', 'obliging', 'neighbouring', 'bored', 'lindley', 'Mind', 'retention', 'Coast', 'Brain', 'plac', 'Pole', 'tan', 'Treatises', 'Crystalline', 'Prolegom', 'Edge', 'centre', 'Surgeon', 'ear', 'old', 'wereS', 'breaking', 'KMA', 'c)f', 'beliese', 'Ol', 'practicable', 'Soldier', 'requisite', 'pullet', 'palpable', 'Collega', 'morassy', 'oftet1', 'Lemma', 'i767', 'shallow', 'Respiration', 'Kidneys', 'enlarge', 'cellsof', 'Spleen', 'optic', 'queen.\\Yhen', 'isame', 'Pegu', 'purging', 'ait', 'wooden', 'swarm', 'tenderer', 'Calder', 'Pots', 'vere', 'Authors', 'trunk', 'accounting', 'comtnon', 'Source', 'Chronology', 'Orchard', 'laid', 'sand', 'acceptable', 'Orthodox', 'perfbet', 'fourel', 'comb', 'ossification', 'electrifying', 'Hospital', 'Sir', 'Rhumbs', 'soutld', 'jealousy', 'imbutus', 'Ship', 'speed', 'lollg', 'deducible', 'lnouth', 'comparative', 'botto', 'Pearl', 'High', 'goth', 'fluid', 'Digestion', 'stratum', 'Breasts', 'Exulcerations', 'Informer', 'oWearly', 'Jnow', 'time', 'proofsyet', 'cup', 'master', 'Channels', 'chronical', 'change', 'marl', 'etltern', 'hot', 'S(3me', 'intervention', '90o', 'proneness', 'sea', 'Ferment', 'proposal', 'il', 'station', 'Socio', 'swarmZ', 'organical', 'arsenic', 'FINI', 'Universal', 'detnorlstration', 'J.', 'early', '-xvhole', 'seat', 'scent', 'Shrubs', 'friendship', 'sudicient', 'MF', 'glan', 'Evelyn', 'nut', 'Villosa', 'wl@at', 'operatlon', 'moor', 'NNE', 'carbonic', 'suggestion', 'Julle', 'immediate', 'thana', 'hox', '-alltumn', 'mechanic', 'airing', 'fir', 'Acute', 'FM', 'bourer', 'watchcap', 'turn', 'Workmanship', 'RS', 'Cape', 'Rati', 'lake', 'Danish', 'si', 'fi11', 'ust', 'Clows', 'succession', 'husband', 'sure', 'Gibson', 'Gowkerhill', 'physiology', 'expertus', 'instant', 'Knotlanes', 'Size', 'hope', 'mode', 'curved', 'Orient', 'immaterial', 'stranger', 'sharpeius', 'quinquesection', 'Natura', 'rrhe', 'Zerdotalia', 'community', 'predecessor', 'capillary', 'difficulty', 'hard', 'dull', 'G/', 'Riodunum', 'Physiological', 'Charas', 'arc', 'Alexandria', 'luminous', 'intoone', 'Preparation', 'lunar', 'Pepys', '2~', 'Cognation', 'rivet', 'occasion', 'Carmine', 'Posture', 'Circuit', 'Aug.', 'amould', 'Coelestis', 'growth', 'RAMSDEN', 'Syllabus', 'postquam', 'few', 'P.R.S.', 'HALLEY', 'section', 'laya', 'like', 'exerement', 'Redi', 'thermometer', 'exert', 'twellty', 'consentaneous', 'Meridians', 'Dog', 'E.N.E.', 'phenomenon', 'delineation', 'beginning', 'cutaneous', 'continuedto', 'establishment', 'shake', 'COD', 'pressure', 'pearl', 'overplus', 'EK', 'Castleton', 'concise', 'setting', 'ark', 'tract', 'Extract', 'Robertson', 'Microscope', 'Meridian', 'ignorance', 'drag', 'History', 'Mancunium', 'enslling', 'Rose', 'Constant', 'distance', 'twostnall', 'instinctive', 'EE', 'C.', 'passage', 'impregnating', 'whetherthe', 'ADC', 'Distances', 'stomacEl', 'yellower', 'Qualem', 'LEMM', 'connection', 'fferent', 'faritla', 'Oxonii', 'computed', 'Care', 'Kitchen', 'cecorlomy', '-period', 'cmimal', 'Skeletons', 'therefrom', 'thetnselve', 'ect', 'Thomas', 'curve', 'recline', 'shepherd', 'thera', 'irldelent', 'plan', 'satisfaction', 'field', 'Instruments', 'Mercator', 'Velocities', 'Parisian', 'Carbuncles', 'lively', 'single', 'incapable', 'Catif', 'spirituous', 'absolute', 'Campanae', 'Burton', 'log', 'expression', 'fve', 'human', 'Italy', 'Cambodunum', 'glandular', 'Culture', '~Burgh', 'Materia', 'Climats', 'margin', 'Claighwait', 'principal', 'QB', 'conducive', 'Franc', 'late', 'Revolution', 'liv', 'map', 'Bengala', 'nester', 'Cutts', 'creme', 'permutation', 'loalX', 'l)reeder', 'hifire', 'acquaintance', 'Sensitive', 'Nourishment', 'power', 'safety', 'absurdity', 'RM', 'Slaighwait', 'comb-', 'Gulf', 'outsvard', 'Volulne', 'ehrysalise', 'aconnexion', 'step', 'Commissions', '51~', 'eld', 'nlost', 'juncture', 'Earthen', 'Compositions', 'illud', 'Overseer', 'geometrical', 'HB', 'bad', 'Close', 'Remedies', 'busilless', 'Thom', 'Itz', 'diffidence', 'weights', 'incommodious', 'respective', 'Ware', 'mov', 'peevish', 'hospital', '31~', 'positiorl', 'utllen', 'investigation', 'Vitae', 'pleasure', 'Energetical', 'HRI', 'Fig', 'Excellent', 'solid', 'Sterne', 'Taume', 'minuti', 'Optic', 'salve', 'assected', 'Northwest', 'quart', 'offience', 'Sight', 'genuine', 'Rules', 'wtorker', 'CAF', 'Diseases', 'offspring', 'Sugar', 'Book', '.he', 'ankle', 't70ung', 'Growth', 'omnibus', 'Sharp', 'soa', 'disorder', 'intelligent', 'rhomboidal', 'decisive', 'ixl', '5E', 'Constable', 'rI', 'Iitna', 'industry', 'BRAHE', 'wet', 'bully5and', 'lmaggot', 'internal', 'ISrst', 'deficiency', 'Goats', 'Black', 'Writers', 'breadth']
```
Observations
    - words with two letters: tv
    - words with numbers, special characters
    - words in all caps
        - how to know if they are names?
        - names are handled separatedly: J., Evelyn
    - acronyms: P.R.S., CAF, 5E, QB
    - stopwords: Sir

Decisions
    - filter out words all in caps -> send to tuple (all caps words group, textId), then apply normal capitalization to the tuple 
    - before tokenization with spacy
        - apply regular expressions to remove words with special characters/numbers but without dividing hyphenated words
        - remove stopwords based on standard lists

2. Oxy-terms presence

Observations: 
- no hits for "potential", even though Aristotles coined the term
    #REVIEW: Connection phlogiston~potential, does it match what Chang said about how phlogiston could have been "electron"?

## Results

Under \results
    - 2 folders for each strategy:
        - wo-past: without texts published in previous years
        - w-past: with texts published in previous years
    - each folder contains a folder for each decade between 1660-1800
    - each decade folder contains
        - clean_corpus: corpus after data cleaning round 2
        - df_[year]: corpus after data cleaning round 1
        - doc_labels: textIds corresponding to the rows in the dtm
        - dtm_sparse: document-term matrix
        - vocab: words corresponding to the columns in the dtm

# Classification
Input: dtm
Metrics: perplexity, topic coherence
Output: M1 (doc x term), M2 (doc x topic), M3 (topic x term) 

- [x] #REVIEW: Select topic modeling approach: tm-toolkit
    - TopicTimelines considers the same #topics per year
    - Topics2Themes uses NMF (efficient for short texts?)
    - [tm-toolkit](https://tmtoolkit.readthedocs.io/en/latest/) uses LDA (efficient for long texts?)
    - Previous works
        - [LDA-topics](https://ids-pub.bsz-bw.de/frontdoor/deliver/index/docId/5474/file/Fankhauser_Knappen_Teich_Topical_Diversification_Over_Time_In_The_Royal_Society_Corpus_2016a.pdf), 30 topics until 1929
- [x] Define evaluation metrics
    - [x] 2 metrics to estimate the optimal number of topics based on this approach: https://asistdl.onlinelibrary.wiley.com/doi/full/10.1002/asi.24533
        1. `metric_held_out_documents_wallach09`-Perplexity: Wallach et al. 2009. Evaluation methods for topic models. --same as ref
        2. `metric_coherence_mimno_2011`-Topic coherence: Mimno et al. 2011. Optimizing semantic coherence in topic models.
    - #REVIEW: Other useful topic metrics: https://arxiv.org/pdf/2005.10125

- [x] Topic model evaluation
    - [x] Define optimal number of topics (best_k) per decade: [6,6,6,6,6,6]
        - [x] Plot metrics from min_topic-max_topic(=30)
        - [x] Define best_k for each decade
- [x] Topic model
    - [x] Get topic words
        - [x] Apply LDA model with best_k for each decade
    - [x] Get topic labels
- [x] Visualization
    - [x] Cosine similarity/JSD heatmaps of topics per decade
        - #REVIEW: Which topics are similar to "oxygen" over time? What are their top words?

#REVIEW: SOTA method for classification of scholarly documents is contrastive learning based on citations. However, in the RSC we don't have a referentiation format during the target centuries. Hence, we will use topics for now to estimate the classes. Future work will address this limitation.

## Observations

- Running the model with max_topics = 100 doesn't make sense since the ref LDA model has a max no. of 30.
- Approaches comparison:
    - w-past: no acid topic
    - wo-past: acid topic
    - #NOTE: the cumulative approach starts from 1750 (papers before that are not considered in the analysis)

## Results

- #REVIEW: our model has symmetric alpha

# Specialization
Input: edge index (A, relationship_type, B)
Output: knowledge graph

- [x] Select graph construction approach: GCNConv from PyTorch Geometric
    - KnowFormer is a GAT, we can find the relevant nodes using attentional weights
    - General TemporalGCN
- #NOTE: Stefania: Measuring paradigmatic change using entropy--high entropy is linked to cluster dynamics/transfer, low entropy is linked to one term driving the paradigm
- #NOTE: Stefania: Alignment of document-level and word-level strategy: there will be a period of stabilization (high entropy-> low entropy)

- [ ] Define important documents
    - Texts that mention