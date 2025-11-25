# On the evolution of scientific concepts

Main task: explain how contextual factors interactions drive conceptual change, why these dynamics evaluate scientific progress and forecast research trends

Subtasks:
1. Localization.
2. Knowledge Entities & Relations Extraction.
3. Knowledge Graph Construction.
4. Knowledge Graph Reasoning.

# Work Environment
- [ ] Python environment: requirements.txt
    - [ ] local: using conda
    - [ ] HPC: using venv #NOTE: might create Docker image later
- [ ] MVP PPT: jupyter notebook
    - [ ] local: create it and send to Stefania

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
		- [ ] Word segmentation
	- [ ] Keyword extraction
		- [ ] Extract word features
		- [ ] Classify whether each word is a keyword
# Keyword graph
- [ ] Keyword graph
	- [ ] Construct keyword graph
		- [ ] Construct or update keyword graph by keyword co-occurrence in new incoming documents
	- [ ] Split keyword graph
		- [ ] Identify changed part of keyword graph
		- [ ] Community detection
		- [ ] Filtering out small sub-graphs
# Cluster events
- [ ] Cluster events
	- [ ] First layer clustering
		- [ ] CLuster new documents by keyword communities
	- [ ] Second layer clustering
		- [ ] Doc-pair relationship classification
		- [ ] Construct doc graph
		- [ ] Community detection on doc graph
# Grow stories
- [ ] Grow stories
	- [ ] Find related story
		- [ ] Identify candidate stories
		- [ ] Find most related story
		- [ ] If no related story, create a new story
	- [ ] Grow story forest
		- [ ] Compare new events with existing story nodes
		- [ ] Merge same events, or insert events to stories

# References
- Story forest from Fig. 21.2 in L. Wu et al. (2022). Chapter21: Graph Neural Networks in Natural Language Processing. In _Graph Neural Networks: Foundations, Frontiers, and Applications_ (pp. 463--481). Springer Nature Singapore. [https://doi.org/10.1007/978-981-16-6054-2](https://doi.org/10.1007/978-981-16-6054-2)
