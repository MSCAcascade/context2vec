Code repository for the paper [Modeling Changing Scientific Concepts with Complex Networks: A Case Study on the Chemical Revolution](https://aclanthology.org/2026.latechclfl-1.14/), accepted at the Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (EACL 2026).

## Summary

Abstract
> While context embeddings produced by LLMs can be used to estimate conceptual change, these representations are often not interpretable nor time-aware. Moreover, bias augmentation in historical data poses a non-trivial risk to researchers in the Digital Humanities. Hence, to model reliable concept trajectories in evolving scholarship, in this work we develop a framework that represents prototypical concepts through complex networks based on topics. Utilizing the Royal Society Corpus, we analyzed two competing theories from the Chemical Revolution (phlogiston vs. oxygen) as a case study to show that onomasiological change is linked to higher entropy and topological density, indicating increased diversity of ideas and connectivity effort.

Inputs: [Royal Society Corpus](https://fedora.clarin-d.uni-saarland.de/rsc_v6/)
Outputs: Zenodo (tba)

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MSCAcascade/context2vec.git
   cd context2vec
   ```

2. **Set up the environment**:
   Ensure you have Python installed. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the analysis**:
   Use the `--task` argument to specify the task you want to execute. Replace `<task>` with one of the following options in the specified order: `data`, `eda`, `tm-eval`, `tm-topics`, `clustering`, `specialization`, or `linking`.

   Example commands:
   - **Data preprocessing**:
     ```bash
     python main.py --task data
     ```
   - **Exploratory Data Analysis (EDA)**:
     ```bash
     python main.py --task eda
     ```
   - **Topic Modeling Evaluation**:
     ```bash
     python main.py --task tm-eval
     ```
   - **Extract Topic Words**:
     ```bash
     python main.py --task tm-topics
     ```
   - **Clustering Analysis**:
     ```bash
     python main.py --task clustering
     ```
   - **Specialization Analysis**:
     ```bash
     python main.py --task specialization
     ```
   - **Linking Analysis**:
     ```bash
     python main.py --task linking
     ```

5. **Outputs**:
   The results will be saved in the `results/` directory (or as specified in the script).

6. **Deactivate the environment** (optional):
   ```bash
   deactivate
   ```

For more details, refer to the paper: [Modeling Changing Scientific Concepts with Complex Networks](https://aclanthology.org/2026.latechclfl-1.14/).

