# On the evolution of scientific concepts

Main task: explain how contextual factors interactions drive conceptual change, why these dynamics evaluate scientific progress and forecast research trends

Subtasks:
1. Localization.
2. Prepare data.
3. Construct graph.
4. Clustering.
5. Grow stories.

# Work Environment
- [ ] Python environment: requirements.txt
    - [ ] local: using conda
    - [ ] HPC: using venv #NOTE: might create Docker image later
- [ ] MVP PPT: jupyter notebook
    - [ ] local: create it and send to Stefania

- [ ] Backup data & results
	- [x] Personal & work laptops using Syncthing
	- [x] OneDrive

(later) for functionality
- [ ] Validate CLI tool execution: argparse, config.py
    - More info: 
        - https://realpython.com/command-line-interfaces-python-argparse/
        - https://codesamplez.com/development/cli-tool-with-python#h-understanding-cli-tool-execution-flow
- [ ] PyTorch Lighting OOP project structure

(later) for reproducibility
- [ ] Input data & results: Zenodo #NOTE: for now OneDrive
- [ ] Code: GitHub #NOTE: main~PhD, branches~subprojects that will have its own repo once published

(later) for interactivity
- [ ] Web application: Django, Vue

# Prepare data
- [ ] Prepare data
	- [ ] Processing documents
		- [ ] Document filtering
			- [x] Define target period
				- RSCV6.0.4: 1665-1920
				- SemanticScholar: 1931-today #TODO: if there's time
			- [x] Define word features: **keywords, events, influencers-influencees**
				#REVIEW What methods extract (dynamic, context-sensitive) word features that increase localization of cultural artifacts? 
				- pointwise KLD
				- surprisal
				- HP cascades
	- [ ] Event processing
		- [ ] Define events
			- [ ] Scientific
				- oxygen
				#TODO how to localize/filter doc/word features for the following scenarios?
				- radiation
				- nebulae
				- evolution
			#REVIEW **For future work**, represent this contextual information so the heterogenous graph trains the node embeddings with it, and the agent can access it
			- [ ] Historical
				- God -> Nature
				- War, gun
				- Famine
			- [ ] Epidemiological
				- Disease, Pox
			#REVIEW List discoveries reported in the RSC
		- [ ] Define keywords & influencers-influencee per event
	- [ ] Keyword processing
		- [ ] Define token segmentation
			- uni-gram
			- bi-gram #TODO: if there's time
		- [ ] Syntactic complexity
			- [ ] Get KLD score
			- [ ] Get **pointwise-KLD scores**
			- [ ] Get surprisal scores
			- [ ] Get **syntagmatic productivity**
		- [ ] Semantic complexity
			- [ ] Get **word embeddings**: word2vec
			- [ ] Get **topics**: LDA
				#TODO Define automatically the optimal number of topics
			- [ ] Get **paradigmatic productivity**
		- [ ] Classify whether each word is a keyword
			#REVIEW Build topic models only with docs~keywords?
	- [ ] Influencers-influencee processing
		- [ ] Get in-text PN & authors per doc
		- [ ] Get author adj. matrix
# Construct graph
- [ ] Keyword co-occurrence
	- [ ] Construct keyword graph
		- [ ] Construct or update keyword graph by keyword co-occurrence in new incoming documents
	- [ ] Split keyword graph
		- [ ] Identify changed part of keyword graph
		- [ ] Community detection
		- [ ] Filtering out small sub-graphs
- [ ] Topic correlation
	- [ ] Get docs adj. matrix based on doc-topic distr. JS distance
	- [ ] Get HAC 
# Clustering
- [ ] Cluster events
	- [ ] Keyword co-occurrence
		- [ ] First layer clustering
			- [ ] Cluster new documents by keyword communities
		- [ ] Second layer clustering
			- [ ] Doc-pair relationship classification
			- [ ] Construct doc graph
			- [ ] Community detection on doc graph
	- [ ] Topic correlation
		- [ ] First layer clustering
			- [ ] Cluster documents in topic using Louvain
			- [ ] Get network metrics of topic
			- [ ] Compare network metrics across topics
				#REVIEW Which topic has the highest entropy? Are the topics with the highest entropy those that evolve?
		- [ ] Second layer clustering
			- [ ] Define how new documents would be processed #TODO if there's time
		#TODO Alternatively, classify the nodes to avoid using LDA-topics
# Grow stories
- [ ] Grow stories
	- [ ] Keyword co-occurrence
		- [ ] Find related story
			- [ ] Identify candidate stories
			- [ ] Find most related story
			- [ ] If no related story, create a new story
		- [ ] Grow story forest
			- [ ] Compare new events with existing story nodes
			- [ ] Merge same events, or insert events to stories
	- [ ] **Link prediction**
		- [ ] Define input data
			- edge-index: node embeddings similarity
			- features matrix: aggregation (word~topics, authors~topics)
				#REVIEW How appropiate is it to use PN as word vectors (e.g., handle names)?
				#REVIEW How to augment/combine author citations vector w/author word2vec -> node2vec?
				#REVIEW Could edge features be built from KLD/surprisal scores?
			- timestamps #TODO if there's time
		- [ ] Train GCN to predict links between graph snapshots
			#REVIEW How does the final graph reflect the observations obtained thus far?
			- [ ] Build Sankey diagram (future->past)
	#TODO build KG-LLM agent for QA: key features in the development of X trajectory, and so on

# References
- Story forest from Fig. 21.2 in L. Wu et al. (2022). Chapter21: Graph Neural Networks in Natural Language Processing. In _Graph Neural Networks: Foundations, Frontiers, and Applications_ (pp. 463--481). Springer Nature Singapore. [https://doi.org/10.1007/978-981-16-6054-2](https://doi.org/10.1007/978-981-16-6054-2)
