import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
from torch_geometric.nn import GCNConv

#import openai
import sys
import os

# Add src to path for utils
sys.path.append('src')
import utils as u

# ------------- Configuration --------------
DOC_TOPIC_PATH_T1 = 'results/wo-past/1770/topics_2_30/doc_topic_distr_6.txt'  # snapshot 1 features
DOC_TOPIC_PATH_T2 = 'results/wo-past/1800/topics_2_30/doc_topic_distr_6.txt'  # snapshot 2 features
DOC_METADATA_PATH = 'results/articles_16to18.csv'        # CSV with: textId, text, decade
IMPORTANT_DOCS_PATH = "src/tasks/specialization/rsc604_text_priestley&lavoisier.txt" # docs mentioning Lavoisier or Priestley

EDGE_SIMILARITY_THRESHOLD = 0.3  # Using JSD threshold (lower = more similar)
HIDDEN_DIM = 32
NUM_EPOCHS = 50
LR = 0.01
TOP_K = 10
LIMIT = None  # Set to None for no limit, or integer for debugging with limited docs
#OPENAI_API_KEY = 'your_openai_api_key'  # Replace with your real OpenAI API key

print("Starting Temporal GNN-LLM Analysis...")
if LIMIT is not None:
    print(f"DEBUG MODE: Using only {LIMIT} documents per snapshot")

# ------------ Load Important Documents List ------------
print("Loading important documents list...")
with open(IMPORTANT_DOCS_PATH, 'r') as f:
    important_docs_list = [line.strip() for line in f.readlines() if line.strip()]

print(f"Loaded {len(important_docs_list)} important document IDs")

# ------------ Load Doc-Topic Matrices with Document IDs ------------
print("Loading doc-topic matrices and document labels...")

# Load document labels to get actual document IDs for each snapshot
doc_labels_t1_raw = u.load_object('results/wo-past/1770/doc_labels.pkl')
doc_labels_t2_raw = u.load_object('results/wo-past/1800/doc_labels.pkl')

# Extract the part after the dash (e.g., 'df_1770-105870' -> '105870')
def extract_doc_id(label):
    if '-' in str(label):
        return str(label).split('-')[-1]
    else:
        return str(label)
    
doc_labels_t1 = [extract_doc_id(label) for label in doc_labels_t1_raw]
doc_labels_t2 = [extract_doc_id(label) for label in doc_labels_t2_raw]

doc_topic_t1 = np.loadtxt(DOC_TOPIC_PATH_T1)
doc_topic_t2 = np.loadtxt(DOC_TOPIC_PATH_T2)

# Apply limit for debugging
if LIMIT is not None:
    print(f"Applying limit of {LIMIT} documents per snapshot...")
    
    # Limit doc-topic matrices
    doc_topic_t1 = doc_topic_t1[:LIMIT]
    doc_topic_t2 = doc_topic_t2[:LIMIT]
    
    # Limit doc labels accordingly
    doc_labels_t1 = doc_labels_t1[:LIMIT]
    doc_labels_t2 = doc_labels_t2[:LIMIT]
    
    print(f"Limited T1: {len(doc_labels_t1)} documents, shape: {doc_topic_t1.shape}")
    print(f"Limited T2: {len(doc_labels_t2)} documents, shape: {doc_topic_t2.shape}")
else:
    print(f"T1: {len(doc_labels_t1)} documents, shape: {doc_topic_t1.shape}")
    print(f"T2: {len(doc_labels_t2)} documents, shape: {doc_topic_t2.shape}")

# Verify alignment
assert len(doc_labels_t1) == doc_topic_t1.shape[0], "T1 doc_labels and matrix size mismatch"
assert len(doc_labels_t2) == doc_topic_t2.shape[0], "T2 doc_labels and matrix size mismatch"

# -------------- Load and Process Metadata ----------------
print("Loading and processing metadata...")

# Load full metadata
docs_df_full = pd.read_csv(DOC_METADATA_PATH)

# Check for required columns and rename if necessary
if 'textId' in docs_df_full.columns:
    docs_df_full = docs_df_full.rename(columns={'textId': 'doc_id'})
elif 'text_id' in docs_df_full.columns:
    docs_df_full = docs_df_full.rename(columns={'text_id': 'doc_id'})
elif 'doc_id' not in docs_df_full.columns:
    raise ValueError("The metadata CSV must contain either 'textId', 'text_id', or 'doc_id' column.")

if 'decade' not in docs_df_full.columns:
    raise ValueError("The metadata CSV must contain a 'decade' column.")

# Extract the part after the dash from doc_id if it has the format 'df_xxxx-yyyyy'
docs_df_full['doc_id'] = docs_df_full['doc_id'].apply(extract_doc_id)

# Drop unnecessary columns
columns_to_drop = ['text', 'article_text'] if 'text' in docs_df_full.columns else ['article_text']
docs_df_full = docs_df_full.drop(columns=[col for col in columns_to_drop if col in docs_df_full.columns])

print(f"Metadata columns: {list(docs_df_full.columns)}")
print(f"Total metadata records: {len(docs_df_full)}")
print(f"Sample doc_ids: {docs_df_full['doc_id'].head().tolist()}")
print(f"Sample decades: {docs_df_full['decade'].head().tolist()}")

def create_snapshot_metadata(doc_labels, snapshot_name):
    """Create metadata DataFrame aligned with doc-topic matrix for a snapshot"""
    # Filter metadata to only include documents in this snapshot
    snapshot_df = docs_df_full[docs_df_full['doc_id'].isin(doc_labels)].copy()
    
    # Reorder to match doc_labels order (critical for matrix alignment)
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_labels)}
    snapshot_df['matrix_idx'] = snapshot_df['doc_id'].map(doc_to_idx)
    snapshot_df = snapshot_df.sort_values('matrix_idx').reset_index(drop=True)
    
    # Verify we have all documents
    missing_docs = set(doc_labels) - set(snapshot_df['doc_id'])
    if missing_docs:
        print(f"Warning: {len(missing_docs)} documents missing from metadata for {snapshot_name}")
        # Create dummy entries for missing documents
        for doc_id in missing_docs:
            # Extract decade from snapshot name
            if snapshot_name == 't1':
                default_decade = 1770
            else:
                default_decade = 1800
                
            dummy_row = {
                'doc_id': doc_id,
                'decade': default_decade,
                'matrix_idx': doc_to_idx[doc_id]
            }
            snapshot_df = pd.concat([snapshot_df, pd.DataFrame([dummy_row])], ignore_index=True)
        snapshot_df = snapshot_df.sort_values('matrix_idx').reset_index(drop=True)
    
    # Create importance labels using the loaded important docs list
    important_set = set(important_docs_list)
    snapshot_df['important'] = snapshot_df['doc_id'].apply(
        lambda x: 1 if x in important_set else 0
    )
    
    # Add document type
    if snapshot_name == 't1':
        snapshot_df['doc_type'] = 'baseline'  # Documents in first snapshot (1770)
    else:  # t2
        snapshot_df['doc_type'] = 'current'  # Documents in current snapshot (1800)
    
    print(f"{snapshot_name} metadata: {len(snapshot_df)} documents")
    print(f"  - Important docs: {snapshot_df['important'].sum()}")
    print(f"  - Decade range: {snapshot_df['decade'].min()} - {snapshot_df['decade'].max()}")
    
    return snapshot_df

# Create aligned metadata for each snapshot
docs_df_t1 = create_snapshot_metadata(doc_labels_t1, 't1')
docs_df_t2 = create_snapshot_metadata(doc_labels_t2, 't2')

# Final verification
assert len(docs_df_t1) == doc_topic_t1.shape[0], "T1 metadata-matrix alignment failed"
assert len(docs_df_t2) == doc_topic_t2.shape[0], "T2 metadata-matrix alignment failed"

# ---------- Prepare Features with Temporal Info ----------
def prepare_features(doc_topic, docs_df, snapshot_name):
    """Prepare node features with normalized temporal information"""
    # Use decade for temporal information
    decades = docs_df['decade'].values
    
    # Normalize decades within this snapshot
    decade_range = decades.max() - decades.min()
    if decade_range > 0:
        decade_norm = (decades - decades.min()) / decade_range
    else:
        decade_norm = np.zeros_like(decades, dtype=float)  # Handle case where all decades are the same
    
    # Convert to tensors
    x = torch.tensor(doc_topic, dtype=torch.float)
    decade_tensor = torch.tensor(decade_norm, dtype=torch.float).unsqueeze(1)
    
    # Concatenate features (doc-topic + normalized decade)
    x_with_decade = torch.cat([x, decade_tensor], dim=1)
    
    # Create labels
    labels = torch.tensor(docs_df['important'].values, dtype=torch.float)
    
    print(f"{snapshot_name} features shape: {x_with_decade.shape}")
    print(f"{snapshot_name} important docs: {labels.sum().item()}")
    
    return x_with_decade, labels, decades

x_t1, labels_t1, decades_t1 = prepare_features(doc_topic_t1, docs_df_t1, 'T1')
x_t2, labels_t2, decades_t2 = prepare_features(doc_topic_t2, docs_df_t2, 'T2')

# ---------- Build Edge Index Function with JSD ----------
def build_edge_index_jsd(doc_topic, decades, threshold=EDGE_SIMILARITY_THRESHOLD):
    """Build edges using Jensen-Shannon Distance (lower = more similar)"""
    print(f"Computing JSD for {doc_topic.shape[0]} documents...")
    
    # Compute pairwise JSD
    jsd_distances = pdist(doc_topic, metric='jensenshannon')
    jsd_matrix = squareform(jsd_distances)
    
    src_nodes, dst_nodes = [], []
    n = jsd_matrix.shape[0]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # For distance: lower values = more similar
            # Also maintain temporal constraint: earlier decades can influence later ones
            if jsd_matrix[i, j] < threshold and decades[i] <= decades[j]:
                src_nodes.append(i)
                dst_nodes.append(j)
    
    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    print(f"Created {edge_index.shape[1]} edges with JSD threshold {threshold}")
    
    return edge_index

print("Building edge indices with Jensen-Shannon Distance...")
edge_index_t1 = build_edge_index_jsd(doc_topic_t1, decades_t1)
edge_index_t2 = build_edge_index_jsd(doc_topic_t2, decades_t2)

# ---------- Cross-snapshot Analysis ----------
print("\nAnalyzing document evolution between snapshots...")

# Find common documents between snapshots (should be none for wo-past)
common_docs = set(doc_labels_t1).intersection(set(doc_labels_t2))
print(f"Common documents between snapshots: {len(common_docs)}")

# For wo-past approach, we're comparing different sets of documents
# T1: 1770 documents only
# T2: 1800 documents only
print(f"T1 (1770) unique documents: {len(set(doc_labels_t1) - common_docs)}")
print(f"T2 (1800) unique documents: {len(set(doc_labels_t2) - common_docs)}")

# ---------- Handle Different Snapshot Sizes ----------
print("Handling different snapshot sizes...")

max_docs = max(x_t1.shape[0], x_t2.shape[0])
feature_dim = x_t1.shape[1]

def pad_snapshot(x, labels, target_size, snapshot_name):
    """Pad smaller snapshot to match larger one"""
    current_size = x.shape[0]
    if current_size < target_size:
        padding_size = target_size - current_size
        x_pad = torch.zeros(padding_size, x.shape[1])
        labels_pad = torch.zeros(padding_size)
        
        x_padded = torch.cat([x, x_pad], dim=0)
        labels_padded = torch.cat([labels, labels_pad], dim=0)
        
        # Create mask for real vs padded nodes
        mask = torch.ones(target_size, dtype=torch.bool)
        mask[current_size:] = False
        
        print(f"{snapshot_name}: Padded from {current_size} to {target_size} docs")
        return x_padded, labels_padded, mask
    else:
        mask = torch.ones(current_size, dtype=torch.bool)
        print(f"{snapshot_name}: No padding needed ({current_size} docs)")
        return x, labels, mask

x_t1_pad, labels_t1_pad, mask_t1 = pad_snapshot(x_t1, labels_t1, max_docs, 'T1')
x_t2_pad, labels_t2_pad, mask_t2 = pad_snapshot(x_t2, labels_t2, max_docs, 'T2')

# Stack for temporal processing
x_seq = torch.stack([x_t1_pad, x_t2_pad])  # shape (2, max_docs, features)

# Use T2 labels and mask for training (most recent snapshot)
target_labels = labels_t2_pad
target_mask = mask_t2

print(f"Final tensor shapes - x_seq: {x_seq.shape}, target_labels: {target_labels.shape}")

# --------- Simplified Temporal GCN Model ----------
class TemporalGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, timesteps=2):
        super(TemporalGCN, self).__init__()
        self.timesteps = timesteps
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, x_seq, edge_indices):
        outputs = []
        
        for t in range(self.timesteps):
            x_t = x_seq[t]
            edge_index_t = edge_indices[t]
            
            # Apply GCN layers
            out = F.relu(self.gcn1(x_t, edge_index_t))
            out = self.dropout(out)
            out = self.gcn2(out, edge_index_t)
            
            outputs.append(out)
        
        # Combine temporal information (simple approach: use final timestep)
        final_output = outputs[-1]
        importance_out = self.classifier(final_output).squeeze()
        
        return importance_out, outputs

model = TemporalGCN(in_channels=feature_dim, hidden_channels=HIDDEN_DIM, timesteps=2)

# -------- Training ----------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCEWithLogitsLoss()

print("Training Temporal GNN...")
print(f"Target docs with importance=1: {target_labels[target_mask].sum().item()}")

if LIMIT is not None:
    print(f"DEBUG MODE: Training with {LIMIT} docs per snapshot")

model.train()
for epoch in range(NUM_EPOCHS):
    optimizer.zero_grad()
    importance_out, _ = model(x_seq, [edge_index_t1, edge_index_t2])
    
    # Only compute loss on real nodes (not padded)
    masked_out = importance_out[target_mask]
    masked_labels = target_labels[target_mask]
    
    loss = criterion(masked_out, masked_labels)
    loss.backward()
    optimizer.step()
    
    # More frequent updates in debug mode
    update_freq = 5 if LIMIT is not None else 10
    if (epoch + 1) % update_freq == 0 or epoch == 0:
        with torch.no_grad():
            preds = (torch.sigmoid(masked_out) >= 0.5).float()
            acc = (preds == masked_labels).sum().item() / masked_labels.size(0)
        print(f"Epoch {epoch+1}: Loss: {loss.item():.4f}, Acc: {acc:.4f}")

# ---------- Enhanced Inference and Analysis ----------
print("\nPerforming inference...")
model.eval()
with torch.no_grad():
    importance_logits, _ = model(x_seq, [edge_index_t1, edge_index_t2])
    importance_probs = torch.sigmoid(importance_logits)

# Get results for T2 (real nodes only)
t2_importance_probs = importance_probs[mask_t2]

# Get top important documents
topk_important = torch.topk(t2_importance_probs, k=min(TOP_K, len(docs_df_t2)))
top_important_docs = docs_df_t2.iloc[topk_important.indices]

print(f"\nTop {len(top_important_docs)} predicted important documents at snapshot 2 (1800):")
print(top_important_docs[['doc_id', 'decade', 'doc_type', 'important']])

# Compare with ground truth
correct_predictions = top_important_docs['important'].sum()
total_important = docs_df_t2['important'].sum()

print(f"\nPrediction Analysis:")
print(f"Correctly identified important docs: {correct_predictions}/{len(top_important_docs)}")
print(f"Total important docs in T2: {total_important}")
print(f"Precision: {correct_predictions/len(top_important_docs):.3f}")
if total_important > 0:
    print(f"Recall: {correct_predictions/total_important:.3f}")

# ---------- Document Evolution Analysis ----------
print("\nDocument Evolution Analysis:")
print(f"1770 snapshot: {len(docs_df_t1)} documents")
print(f"1800 snapshot: {len(docs_df_t2)} documents")

# Analyze importance distribution across decades
print(f"\nImportance distribution:")
print(f"1770 important docs: {docs_df_t1['important'].sum()}/{len(docs_df_t1)}")
print(f"1800 important docs: {docs_df_t2['important'].sum()}/{len(docs_df_t2)}")

# # ---------------- Enhanced LLM Analysis -----------------
# openai.api_key = OPENAI_API_KEY

# # Prepare document information for LLM
# important_docs_info = '\n'.join([
#     f"- DocID: {row.doc_id}, Decade: {row.decade}, Type: {row.doc_type}, Actual Important: {bool(row.important)}"
#     for _, row in top_important_docs.iterrows()
# ])

# enhanced_prompt = f"""
# You are analyzing a scientific paradigm shift between 1770 and 1800. The analysis compares documents from these two distinct decades using topic modeling.

# CONTEXT: 
# - Snapshot 1: Documents from 1770 decade only
# - Snapshot 2: Documents from 1800 decade only  
# - This represents a 30-year evolution in scientific discourse

# TOP PREDICTED IMPORTANT DOCUMENTS FROM 1800:
# {important_docs_info}

# Please analyze:
# 1. What scientific developments do the 1800 documents suggest compared to 1770?
# 2. How might the topics and themes have evolved over this 30-year period?
# 3. What does this tell us about the paradigm shift in chemistry/science during this era?
# 4. Are there patterns in the predicted important documents that align with known historical developments?
# 5. How do the document topics reflect the transition from phlogiston theory to modern chemistry?

# Note: Document analysis is based on topic modeling features and temporal relationships.
# """

# print("\nRequesting analysis from LLM...")
# try:
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": enhanced_prompt}],
#         max_tokens=500,
#         temperature=0.7,
#     )
#     explanation = response['choices'][0]['message']['content']
#     print("\nLLM Analysis:\n", explanation)
# except Exception as e:
#     print("OpenAI API request failed:", e)
#     print("Skipping LLM explanation.")

# print("\n" + "="*50)
# print("Analysis Complete!")
# print("="*50)

# # Summary statistics
# print(f"Snapshot 1 (1770): {len(docs_df_t1)} documents")
# print(f"Snapshot 2 (1800): {len(docs_df_t2)} documents")
# print(f"Important documents identified in training: {len(important_docs_list)}")
# print(f"Top predicted important documents: {len(top_important_docs)}")
# print(f"Model accuracy on predictions: {correct_predictions}/{len(top_important_docs)} = {correct_predictions/len(top_important_docs):.1%}")

if LIMIT is not None:
    print(f"\n[DEBUG MODE] Limited to {LIMIT} documents per snapshot for faster testing")
    print("Set LIMIT = None for full dataset analysis")