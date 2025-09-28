import umap
import hdbscan
import pandas as pd
import numpy as np
from collections import Counter

# Load your data
df = pd.read_parquet("embedded_queries.parquet")
embeddings_array = np.array(df['embedding'].tolist())

# Step 1: Reduce dimensions with UMAP 
umap_reducer = umap.UMAP(
    n_components=50, 
    random_state=42, 
    n_neighbors=30,        
    min_dist=0.05          
)
reduced_embeddings = umap_reducer.fit_transform(embeddings_array)

# Step 2: Cluster with HDBSCAN 
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,        
    min_samples=10,             
    cluster_selection_epsilon=0.3,  
    metric='euclidean'
)
cluster_labels = clusterer.fit_predict(reduced_embeddings)

df['cluster'] = cluster_labels

FILLER_WORDS = {'the', 'a', 'without', 'and', 'for', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'is', 'it', 'that', 'this', 'are', 'was', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them'}

# Step 3: Analyze clusters
unique_clusters = np.unique(cluster_labels)
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
print(f"Total clusters: {num_clusters}")
print(f"Noise points: {len(df[df['cluster'] == -1])}")

print("\nCluster analysis (excluding noise):")
for cluster_id in sorted(np.unique(cluster_labels)):
    if cluster_id == -1:
        continue
        
    cluster_queries = df[df['cluster'] == cluster_id]['query']
    
    # Get meaningful words
    all_words = ' '.join(cluster_queries).lower().split()
    meaningful_words = [word for word in all_words 
                       if word not in FILLER_WORDS and len(word) > 2]
    
    common_words = Counter(meaningful_words).most_common(3)
    
    if common_words:
        best_label = common_words[0][0]
        cluster_size = len(cluster_queries)
        print(f"Cluster {cluster_id} ({cluster_size} queries): {best_label}")