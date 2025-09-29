
# Customer Search Query Clustering Using Machine Learning

This project applies **machine learning and natural language processing** techniques to analyze and cluster multilingual customer search queries.  
The goal is to provide **insights into customer demand patterns** by leveraging **Sentence-BERT embeddings**, **clustering algorithms**, and an **interactive Streamlit dashboard**.

---

## 🚀 Project Overview
- **Dataset**: Amazon ESCI Challenge (277k shopping queries in English, Japanese, and Spanish).
- **ETL Pipeline**:
  - Download dataset
  - Clean and preprocess queries
  - Translate non-English queries to English
  - Save final dataset for downstream ML tasks
- **Embeddings**: Sentence-BERT (`all-MiniLM-L6-v2`) from HuggingFace, producing 384-dimensional dense vectors.
- **Clustering**:
  - **HDBSCAN** for fine-grained clusters
  - **K-Means** for higher-level super clusters
- **Visualization**:
  - Streamlit dashboard with query distribution charts, cluster exploration, word clouds, and hierarchical visualizations.

---

## 📂 Repository Structure
```bash
  
  ├── download_dataset.py       # Download the dataset
  ├── data_preprocessing.py     # Clean and translate queries
  ├── compare.py                # Comparative analysis (before vs. after preprocessing)
  ├── load_model.py             # Generate embeddings with Sentence-BERT
  ├── clustering.py             # Perform clustering (HDBSCAN + KMeans)
  ├── experimenting.py          # Streamlit dashboard (final output)
  └── README.md                 # Project documentation

````

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-query-clustering.git
   cd customer-query-clustering

---
## ▶️ Running the Pipeline

Run the following scripts in sequence:

1. **Download dataset**

   ```bash
   python download_dataset.py
   ```

2. **Preprocess and translate dataset**

   ```bash
   python data_preprocessing.py
   ```

3. **Compare original vs. cleaned dataset**

   ```bash
   python compare.py
   ```

4. **Generate embeddings**

   ```bash
   python load_model.py
   ```

5. **Perform clustering**

   ```bash
   python clustering.py
   ```

6. **Launch interactive dashboard**

   ```bash
   streamlit run experimenting.py
   ```

---

## 📊 Dashboard Features

The dashboard enables **interactive exploration** of query clusters:

* **Overview**: Metrics, query distribution, cluster size histogram
* **Super Clusters**: Interactive tables, sample queries, sub-cluster pie charts
* **Sub Clusters**: Heatmaps and cluster relationship analysis
* **Word Clouds**: Global, cluster-specific, and comparison word clouds
* **Treemap + Bar Charts**: Hierarchical visualization of clusters


---

## 👩‍💻 Author

**Saja Moussa**
IT & BA student | ML & AI Enthusiast


