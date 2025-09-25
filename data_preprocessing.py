import pandas as pd
from translation import translate
import time
import os


#1:extract
def extract_data(file_path="dataset.csv"):
    print("Extracting data...")
    df = pd.read_csv(file_path)
    print(f"Extracted {len(df)} rows")
    return df


#2:transform
def transform_data(df):
    # Clean data by removing columns and duplicates
    print("Transforming data...")
    
    # Remove columns if they exist
    df = df.drop(columns=["product_id", "example_id"], errors="ignore")
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    print(f"Transformed data: {len(df)} rows remaining")
    return df


#translation
def translate_query(row):
    try:
        translated = translate(row["query"], row["query_locale"])
        return translated
    except Exception as e:
        print(f"Error translating query: {row['query']}, Error: {e}")
        return row["query"]


#3: load
def load_data_incremental(df, output_file="translated_queries.csv", delay=1.0):
    # Save data row by row with translation
    print("Loading data...")
    
    # Remove old file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
        print("Removed existing output file")
    
    # Process each row and save to CSV
    for idx, row in df.iterrows():
        translated_query = translate_query(row)
        
        # Add row to CSV file
        pd.DataFrame([{"query": translated_query}]).to_csv(
            output_file,
            mode="a",
            header=not os.path.exists(output_file),
            index=False
        )
        
        # Wait between requests to avoid hitting rate limits
        time.sleep(delay)
        
        # Show progress every 10 rows
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")
    
    print(f"Translation finished. Saved to {output_file}")
    return output_file


#4: remove duplicates from translated dataset
def remove_translated_duplicates(input_file="translated_queries.csv", output_file="cleaned_translated_queries.csv"):
    # Remove duplicates from the translated dataset
    print("Removing duplicates from translated dataset...")
    
    # Read the translated data
    df = pd.read_csv(input_file)
    print(f"Original translated dataset: {len(df)} rows")
    
    # Remove duplicate queries
    df_clean = df.drop_duplicates(subset=['query'])
    
    # Save cleaned data
    df_clean.to_csv(output_file, index=False)
    
    print(f"Removed {len(df) - len(df_clean)} duplicate queries")
    print(f"Cleaned dataset: {len(df_clean)} rows remaining")
    print(f"Saved cleaned data to {output_file}")
    
    return output_file
def lowercase_queries(input_file="translated_queries.csv", output_file="lowercase_translated_queries.csv"):
    # Convert all queries to lowercase
    print("Converting queries to lowercase...")
    
    # Read the translated data
    df = pd.read_csv(input_file)
    print(f"Original dataset: {len(df)} rows")
    
    # Convert all queries to lowercase
    df['query'] = df['query'].str.lower()
    
    # Save lowercase data
    df.to_csv(output_file, index=False)
    
    print(f"Lowercased all queries")
    print(f"Saved lowercase data to {output_file}")
    
    return output_file

def main():
    # Step 1: Extract original data
    df = extract_data("dataset.csv")
    
    # Step 2: Transform original data
    df = transform_data(df)
    
    # Step 3: Translate and save
    load_data_incremental(df, "translated_queries.csv", delay=1.0)
    
    # Step 4: Remove duplicates from translated data
    remove_translated_duplicates("translated_queries.csv", "translated_queries.csv")
    
    # Step 5: Convert to lowercase
    lowercase_queries("translated_queries.csv", "translated_queries.csv")

if __name__ == "__main__":
    main()