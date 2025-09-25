from sentence_transformers import SentenceTransformer
import pandas as pd
import os

def load_model_once():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

model = load_model_once()

def embedding_dataset(input_file="translated_queries.csv", output_file="embedded_queries.parquet"):
    # Check if embeddings already exist
    if os.path.exists(output_file):
        print(f"Embeddings already exist at {output_file}. Skipping generation.")
        return pd.read_parquet(output_file)
    
    df = pd.read_csv(input_file)
    
    if 'query' not in df.columns:
        raise ValueError("CSV file must contain a 'query' column")
    
    # Add index column to track position
    df = df.reset_index().rename(columns={'index': 'query_id'})
    df['query_id'] = df['query_id'] + 1
    
    print(f"Processing {len(df)} queries...")
    
    embeddings = model.encode(df['query'].tolist(), show_progress_bar=True)
    df['embedding'] = embeddings.tolist()
    df = df[['query_id', 'query', 'embedding']]
    
    df.to_parquet(output_file, index=False)  
    print(f"Saved {len(df)} queries to Parquet: {output_file}")
    
    return df

def show_parquet_sample(parquet_file="embedded_queries.parquet", sample_size=5):
    """Display a sample of the Parquet data"""
    df = pd.read_parquet(parquet_file)
    
    print(f"\n{'='*60}")
    print(f"PARQUET FILE SAMPLE (showing {sample_size} of {len(df)} queries):")
    print(f"{'='*60}")
    
    # Show query_id and query columns
    sample_df = df[['query_id', 'query']].head(sample_size)
    print(sample_df.to_string(index=False))
    
    # Show embedding dimension info
    print(f"\nEmbedding dimension: {len(df['embedding'].iloc[0])}")
    print(f"First embedding sample (first 5 values): {df['embedding'].iloc[0][:5]}...")
    
    return df

if __name__ == "__main__":  
    # Generate or load embeddings
    df = embedding_dataset(input_file="translated_queries.csv", output_file="embedded_queries.parquet")
    
    # Show sample of the data
    show_parquet_sample("embedded_queries.parquet", sample_size=8)