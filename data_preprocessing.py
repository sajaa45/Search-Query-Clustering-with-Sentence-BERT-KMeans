import pandas as pd
from translation import translate
import time
import os


#3:extract
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



def main():
     df = extract_data("dataset.csv")
     df = transform_data(df)
     load_data_incremental(df, "translated_queries.csv", delay=1.0)
    

if __name__ == "__main__":
    main()