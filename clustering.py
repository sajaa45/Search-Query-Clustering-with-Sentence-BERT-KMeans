import pandas as pd
import numpy as np

df = pd.read_parquet("embedded_queries.parquet")
embeddings_array = np.array(df['embedding'].tolist())

print(f"\nDataFrame shape: {df.shape}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"Embeddings array shape: {embeddings_array.shape}")


print(f"\nFirst embedding (first 10 values): {embeddings_array[0][:10]}...")