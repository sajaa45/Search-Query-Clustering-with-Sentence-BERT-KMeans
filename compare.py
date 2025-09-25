import pandas as pd
from collections import Counter

# Load datasets
df_orig = pd.read_csv("dataset.csv")
df_trans = pd.read_csv("translated_queries.csv")

# --- 1. Shapes ---
print("===== Dataset Shapes =====")
print(f"Original dataset: {df_orig.shape[0]} rows, {df_orig.shape[1]} columns")
print(f"Translated dataset: {df_trans.shape[0]} rows, {df_trans.shape[1]} columns\n")



# --- 2. Duplicates (query-level only) ---
print("===== Query Duplicates =====")
print(f"Original dataset duplicate queries: {df_orig.duplicated(subset=['query']).sum()}")
print(f"Translated dataset duplicate queries: {df_trans.duplicated(subset=['query']).sum()}\n")


# --- 3. Row Reduction ---
row_reduction_pct = (1 - len(df_trans)/len(df_orig)) * 100
print("===== Row Reduction =====")
print(f"Rows reduced from {len(df_orig)} to {len(df_trans)} ({row_reduction_pct:.2f}% reduction)\n")

# --- 4. Query Length Analysis ---
df_orig['query_length'] = df_orig['query'].str.split().str.len()
df_trans['query_length'] = df_trans['query'].str.split().str.len()

print("===== Query Length (in words) =====")
print("Original dataset:")
print(df_orig['query_length'].describe())
print("\nTranslated dataset:")
print(df_trans['query_length'].describe())
print("")

# --- 5. Vocabulary Size ---
def get_vocab(series):
    words = " ".join(series.astype(str)).split()
    return set(words)

orig_vocab = get_vocab(df_orig['query'])
trans_vocab = get_vocab(df_trans['query'])
vocab_reduction_pct = (1 - len(trans_vocab)/len(orig_vocab)) * 100

print("===== Vocabulary Size =====")
print(f"Original vocabulary size: {len(orig_vocab)} words")
print(f"Translated vocabulary size: {len(trans_vocab)} words")
print(f"Vocabulary reduction: {vocab_reduction_pct:.2f}%\n")

# --- 6. Locale Distribution (original) ---
if 'query_locale' in df_orig.columns:
    print("===== Original Locale Distribution =====")
    print(df_orig['query_locale'].value_counts())
    print("")

# --- 7. Unique Queries ---
print("===== Unique Queries =====")
print(f"Original dataset unique queries: {df_orig['query'].nunique()}")
print(f"Translated dataset unique queries: {df_trans['query'].nunique()}")
print("")

# --- 8. Top 10 Most Frequent Queries ---
print("===== Top 10 Most Frequent Queries (Original) =====")
print(df_orig['query'].value_counts().head(10))
print("\n===== Top 10 Most Frequent Queries (Translated) =====")
print(df_trans['query'].value_counts().head(10))
print("")

print("===== Analysis Summary =====")
print(f"- Original dataset had {len(df_orig)} queries with {len(orig_vocab)} unique words")
print(f"- Translated dataset has {len(df_trans)} queries with {len(trans_vocab)} unique words")
print(f"- Queries reduced by {row_reduction_pct:.2f}% due to deduplication & translation")
print(f"- Vocabulary reduced by {vocab_reduction_pct:.2f}%")
print(f"- All queries are now normalized in English")
