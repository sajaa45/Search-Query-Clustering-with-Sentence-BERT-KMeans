import umap
import hdbscan
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans 

# Load data
df = pd.read_parquet("embedded_queries.parquet")
embeddings_array = np.array(df['embedding'].tolist())

# Reduce dimensions with UMAP 
umap_reducer = umap.UMAP(
    n_components=50, 
    random_state=42, 
    n_neighbors=30,        
    min_dist=0.05          
)
reduced_embeddings = umap_reducer.fit_transform(embeddings_array)

# Cluster with HDBSCAN 
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,        
    min_samples=10,             
    cluster_selection_epsilon=0.3,  
    metric='euclidean'
)
cluster_labels = clusterer.fit_predict(reduced_embeddings)

df['cluster'] = cluster_labels

FILLER_WORDS = {'the', 'a', 'without', 'and', 'no', 'for', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'is', 'it', 'that', 'this', 'are', 'was', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them'}

# Analyze clusters
unique_clusters = np.unique(cluster_labels)
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
print(f"Total clusters: {num_clusters}")
print(f"Noise points: {len(df[df['cluster'] == -1])}")

print("\nCluster analysis (excluding noise):")
for cluster_id in sorted(np.unique(cluster_labels)):
    if cluster_id == -1:
        continue
        
    cluster_queries = df[df['cluster'] == cluster_id]['query']
    
    all_words = ' '.join(cluster_queries).lower().split()
    meaningful_words = [word for word in all_words 
                       if word not in FILLER_WORDS and len(word) > 2]
    
    common_words = Counter(meaningful_words).most_common(3)
    
    if common_words:
        best_label = common_words[0][0]
        cluster_size = len(cluster_queries)
        print(f"Cluster {cluster_id} ({cluster_size} queries): {best_label}")

# Create super clusters using K-Means
def kmeans_super_clusters(df, n_super_clusters=20):
    cluster_centroids = []
    valid_clusters = []
    
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            continue
            
        cluster_embeddings = df[df['cluster'] == cluster_id]['embedding'].tolist()
        centroid = np.mean(cluster_embeddings, axis=0)
        cluster_centroids.append(centroid)
        valid_clusters.append(cluster_id)
    
    if len(cluster_centroids) < n_super_clusters:
        print(f"Warning: Only {len(cluster_centroids)} valid clusters available, but requested {n_super_clusters} super-clusters")
        n_super_clusters = len(cluster_centroids)
    
    cluster_centroids = np.array(cluster_centroids)
    
    kmeans = KMeans(
        n_clusters=n_super_clusters,
        random_state=42,
        n_init=10
    )
    super_labels = kmeans.fit_predict(cluster_centroids)
    
    cluster_to_super = dict(zip(valid_clusters, super_labels))
    df['super_cluster'] = df['cluster'].map(cluster_to_super)
    df['super_cluster'] = df['super_cluster'].fillna(-1) 
    
    return df, cluster_to_super

df, super_mapping = kmeans_super_clusters(df, n_super_clusters=20)

# Generate meaningful single-word names for super-clusters
def generate_super_cluster_names(df, filler_words=FILLER_WORDS):
    super_cluster_names = {}
    
    for super_id in sorted(df['super_cluster'].unique()):
        if super_id == -1:
            super_cluster_names[super_id] = "Noise"
            continue
            
        super_queries = df[df['super_cluster'] == super_id]['query']
        
        all_words = ' '.join(super_queries).lower().split()
        meaningful_words = [
            word for word in all_words 
            if (word not in filler_words and 
                len(word) > 2 and 
                word.isalpha() and
                not word.isdigit())
        ]
        
        word_counts = Counter(meaningful_words)
        
        # Find the most meaningful word (not just highest count)
        for word, count in word_counts.most_common():
            if len(word) > 3 and count > 2:  # More selective criteria
                super_cluster_names[super_id] = word.capitalize()
                break
        else:
            # Fallback if no suitable word found
            if word_counts:
                super_cluster_names[super_id] = word_counts.most_common(1)[0][0].capitalize()
            else:
                super_cluster_names[super_id] = f"Cluster_{super_id}"
    
    return super_cluster_names

def analyze_super_clusters_with_names(df, super_cluster_names):
    print("\nSUPER-CLUSTER ANALYSIS")
    print("=" * 60)
    
    for super_id in sorted(df['super_cluster'].unique()):
        if super_id == -1:
            continue
            
        super_queries = df[df['super_cluster'] == super_id]['query']
        sub_clusters = df[df['super_cluster'] == super_id]['cluster'].unique()
        
        all_words = ' '.join(super_queries).lower().split()
        meaningful_words = [
            word for word in all_words 
            if (word not in FILLER_WORDS and 
                len(word) > 2 and 
                word.isalpha())
        ]
        
        common_words = Counter(meaningful_words).most_common(5)
        
        super_size = len(super_queries)
        super_name = super_cluster_names[super_id]
        
        print(f"\n{super_name} (ID: {super_id})")
        print(f"Size: {super_size} queries")
        print(f"Contains {len(sub_clusters)} sub-clusters: {sorted(sub_clusters)}")
        print(f"Top keywords: {[word for word, count in common_words]}")
        
        # Show sample queries
        print(f"Sample queries:")
        samples_shown = 0
        for sub_cluster in sub_clusters[:3]:  
            sub_cluster_queries = df[
                (df['super_cluster'] == super_id) & 
                (df['cluster'] == sub_cluster)
            ]['query'].head(2)
            for query in sub_cluster_queries:
                if samples_shown < 4:  
                    print(f"  • {query}")
                    samples_shown += 1

# Generate names for super-clusters
super_cluster_names = generate_super_cluster_names(df)

# Apply names to dataframe
df['super_cluster_name'] = df['super_cluster'].map(super_cluster_names)

# Run analysis
analyze_super_clusters_with_names(df, super_cluster_names)

# Create summary table with sub-cluster keywords
def create_super_cluster_summary(df):
    summary_data = []
    
    for super_id in sorted(df['super_cluster'].unique()):
        if super_id == -1:
            continue
            
        super_data = df[df['super_cluster'] == super_id]
        super_name = super_data['super_cluster_name'].iloc[0]
        sub_clusters = sorted(super_data['cluster'].unique())
        
        # Get keywords for each sub-cluster
        sub_cluster_keywords = []
        for sub_cluster_id in sub_clusters:
            sub_cluster_queries = df[df['cluster'] == sub_cluster_id]['query']
            all_words = ' '.join(sub_cluster_queries).lower().split()
            meaningful_words = [word for word in all_words 
                               if word not in FILLER_WORDS and len(word) > 2]
            
            if meaningful_words:
                common_words = Counter(meaningful_words).most_common(1)
                keyword = common_words[0][0] if common_words else f"cluster_{sub_cluster_id}"
                sub_cluster_keywords.append(keyword)
            else:
                sub_cluster_keywords.append(f"cluster_{sub_cluster_id}")
        
        summary_data.append({
            'Super_Cluster_ID': super_id,
            'Super_Cluster_Name': super_name,
            'Total_Queries': len(super_data),
            'Sub_Clusters_Count': len(sub_clusters),
            'Sub_Clusters': sub_cluster_keywords
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Generate and display summary
summary_df = create_super_cluster_summary(df)
print("\nSUPER-CLUSTER SUMMARY TABLE")
print("=" * 60)
print(summary_df.to_string(index=False))

# Save results
df.to_parquet("hierarchical_clustered_queries_with_names.parquet", index=False)
summary_df.to_csv("super_cluster_summary.csv", index=False)

print(f"\nResults saved with {len(super_cluster_names)} named super-clusters!")