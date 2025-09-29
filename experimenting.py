import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Query Clustering Dashboard",
    page_icon="📊",
    layout="wide"
)

# Cache data loading
@st.cache_data
def load_data():
    try:
        df = pd.read_parquet("hierarchical_clustered_queries_with_names.parquet")
        summary_df = pd.read_csv("super_cluster_summary.csv")
        return df, summary_df
    except FileNotFoundError:
        st.error("Data files not found. Please run the clustering script first.")
        st.stop()

# Load data
df, summary_df = load_data()

# Title and description
st.title("📊 Query Clustering Analysis Dashboard")
st.markdown("Interactive visualization of hierarchical query clustering results")

# Sidebar filters
st.sidebar.header("Filters")

# Filter by super cluster
super_clusters = ['All'] + sorted(df['super_cluster_name'].unique())
selected_super_cluster = st.sidebar.selectbox("Select Super Cluster:", super_clusters)

# Filter data based on selection
if selected_super_cluster != 'All':
    filtered_df = df[df['super_cluster_name'] == selected_super_cluster]
else:
    filtered_df = df

# Main metrics (update based on filter)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Queries", len(filtered_df))

with col2:
    if selected_super_cluster == 'All':
        st.metric("Super Clusters", len(filtered_df['super_cluster_name'].unique()))
    else:
        st.metric("Selected Cluster Size", len(filtered_df))

with col3:
    st.metric("Sub Clusters", len(filtered_df['cluster'].unique()))

with col4:
    noise_count = len(filtered_df[filtered_df['cluster'] == -1])
    st.metric("Noise Points", noise_count)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "📊 Super Clusters", "🔍 Sub Clusters", "☁️ Word Clouds"])

with tab1:
    st.header("Clustering Overview")
    
    # Show current filter status
    if selected_super_cluster != 'All':
        st.info(f"📊 Showing results for: **{selected_super_cluster}**")
    else:
        st.info("📊 Showing results for: **All Super Clusters**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Super cluster distribution (use filtered data)
        if selected_super_cluster == 'All':
            super_cluster_counts = df.groupby('super_cluster_name').size().reset_index(name='count')
            title = "Query Distribution by Super Cluster"
        else:
            # Show sub-cluster breakdown for selected super cluster
            sub_cluster_counts = filtered_df.groupby('cluster').size().reset_index(name='count')
            sub_cluster_counts = sub_cluster_counts[sub_cluster_counts['cluster'] != -1]
            sub_cluster_counts['cluster_name'] = sub_cluster_counts['cluster'].apply(lambda x: f"Sub-cluster {x}")
            super_cluster_counts = sub_cluster_counts[['cluster_name', 'count']].rename(columns={'cluster_name': 'super_cluster_name'})
            title = f"Sub-cluster Distribution in {selected_super_cluster}"
        
        super_cluster_counts = super_cluster_counts.sort_values('count', ascending=True)
        
        fig_super = px.bar(
            super_cluster_counts,
            x='count',
            y='super_cluster_name',
            orientation='h',
            title=title,
            labels={'count': 'Number of Queries', 'super_cluster_name': 'Cluster'},
            color='count',
            color_continuous_scale='viridis'
        )
        fig_super.update_layout(height=600)
        st.plotly_chart(fig_super, use_container_width=True)
    
    with col2:
        # Cluster size distribution (use filtered data)
        cluster_sizes = filtered_df[filtered_df['cluster'] != -1].groupby('cluster').size()
        
        if len(cluster_sizes) > 0:
            fig_hist = px.histogram(
                x=cluster_sizes.values,
                nbins=min(20, len(cluster_sizes)),
                title="Distribution of Cluster Sizes",
                labels={'x': 'Cluster Size', 'y': 'Number of Clusters'}
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Summary stats (use filtered data)
            st.subheader("Cluster Statistics")
            stats_df = pd.DataFrame({
                'Metric': ['Mean Size', 'Median Size', 'Min Size', 'Max Size', 'Std Dev'],
                'Value': [
                    f"{cluster_sizes.mean():.1f}",
                    f"{cluster_sizes.median():.1f}",
                    f"{cluster_sizes.min()}",
                    f"{cluster_sizes.max()}",
                    f"{cluster_sizes.std():.1f}"
                ]
            })
            st.table(stats_df)
        else:
            st.info("No valid clusters in the filtered data.")

with tab2:
    st.header("Super Cluster Analysis")
    
    # Summary table
    st.subheader("Super Cluster Summary")
    st.dataframe(summary_df, use_container_width=True)
    
    # Interactive super cluster details
    st.subheader("Detailed View")
    
    selected_cluster = st.selectbox(
        "Select a Super Cluster for details:",
        options=df['super_cluster_name'].unique()
    )
    
    if selected_cluster:
        cluster_data = df[df['super_cluster_name'] == selected_cluster]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Show sample queries
            st.write(f"**Sample queries from {selected_cluster}:**")
            sample_queries = cluster_data['query'].head(10).tolist()
            for i, query in enumerate(sample_queries, 1):
                st.write(f"{i}. {query}")
        
        with col2:
            # Sub-cluster breakdown
            sub_clusters = cluster_data['cluster'].value_counts()
            
            fig_pie = px.pie(
                values=sub_clusters.values,
                names=[f"Sub-cluster {idx}" for idx in sub_clusters.index],
                title=f"Sub-clusters in {selected_cluster}"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    st.header("Sub Cluster Analysis")
    
    # Show current filter status
    if selected_super_cluster != 'All':
        st.info(f"🔍 Showing sub-cluster analysis for: **{selected_super_cluster}**")
    else:
        st.info("🔍 Showing sub-cluster analysis for: **All Super Clusters**")
    
    # Create sub-cluster data based on filter
    if selected_super_cluster == 'All':
        # Show all super clusters and their sub-clusters
        sub_cluster_data = []
        for super_name in df['super_cluster_name'].unique():
            super_data = df[df['super_cluster_name'] == super_name]
            sub_clusters = super_data[super_data['cluster'] != -1]['cluster'].value_counts()
            
            for cluster_id, size in sub_clusters.items():
                sub_cluster_data.append({
                    'Super_Cluster': super_name,
                    'Sub_Cluster_ID': f"Sub-{cluster_id}",
                    'Size': size
                })
        
        if sub_cluster_data:
            sub_df = pd.DataFrame(sub_cluster_data)
            
            # Create a more manageable heatmap
            # Group by super cluster and show top sub-clusters
            fig_treemap = px.treemap(
                sub_df,
                path=['Super_Cluster', 'Sub_Cluster_ID'],
                values='Size',
                title="Sub-cluster Distribution Across Super Clusters (Treemap)",
                color='Size',
                color_continuous_scale='viridis'
            )
            fig_treemap.update_layout(height=600)
            st.plotly_chart(fig_treemap, use_container_width=True)
            
            # Alternative: Bar chart of sub-clusters by super cluster
            col1, col2 = st.columns(2)
            
            with col1:
                # Top sub-clusters overall
                top_subclusters = sub_df.nlargest(15, 'Size')
                fig_bar = px.bar(
                    top_subclusters,
                    x='Size',
                    y='Sub_Cluster_ID',
                    color='Super_Cluster',
                    orientation='h',
                    title="Top 15 Sub-clusters by Size"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # Sub-cluster count by super cluster
                super_subcluster_count = sub_df.groupby('Super_Cluster').size().reset_index(name='Count')
                fig_count = px.bar(
                    super_subcluster_count,
                    x='Count',
                    y='Super_Cluster',
                    orientation='h',
                    title="Number of Sub-clusters per Super Cluster"
                )
                st.plotly_chart(fig_count, use_container_width=True)
    
    else:
        # Show detailed analysis for selected super cluster
        cluster_data = filtered_df[filtered_df['cluster'] != -1]
        
        if len(cluster_data) > 0:
            sub_cluster_sizes = cluster_data['cluster'].value_counts().reset_index()
            sub_cluster_sizes.columns = ['Sub_Cluster_ID', 'Size']
            sub_cluster_sizes['Sub_Cluster_Name'] = sub_cluster_sizes['Sub_Cluster_ID'].apply(lambda x: f"Sub-cluster {x}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart of sub-clusters
                fig_sub = px.bar(
                    sub_cluster_sizes,
                    x='Size',
                    y='Sub_Cluster_Name',
                    orientation='h',
                    title=f"Sub-clusters in {selected_super_cluster}",
                    color='Size',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_sub, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(
                    sub_cluster_sizes,
                    values='Size',
                    names='Sub_Cluster_Name',
                    title=f"Sub-cluster Distribution in {selected_super_cluster}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Show detailed breakdown
            st.subheader("Detailed Sub-cluster Information")
            for _, row in sub_cluster_sizes.iterrows():
                with st.expander(f"{row['Sub_Cluster_Name']} ({row['Size']} queries)"):
                    sub_queries = cluster_data[cluster_data['cluster'] == row['Sub_Cluster_ID']]['query'].head(5)
                    st.write("Sample queries:")
                    for i, query in enumerate(sub_queries, 1):
                        st.write(f"{i}. {query}")
        else:
            st.warning("No valid sub-clusters found in the selected super cluster.")

with tab4:
    st.header("Word Clouds")
    
    # Show current filter status
    if selected_super_cluster != 'All':
        st.info(f"☁️ Word clouds will use data from: **{selected_super_cluster}**")
    
    # Word cloud options
    if selected_super_cluster == 'All':
        wordcloud_option = st.radio(
            "Generate word cloud for:",
            ["All Queries", "Selected Super Cluster", "Compare Super Clusters"]
        )
    else:
        wordcloud_option = st.radio(
            "Generate word cloud for:",
            ["Current Selection", "Compare with Another Cluster"]
        )
    
    FILLER_WORDS = {'the', 'a', 'without', 'and', 'no', 'for', 'with', 'to', 'of', 'in', 'on', 'at', 'by', 'is', 'it', 'that', 'this', 'are', 'was', 'i', 'you', 'he', 'she', 'we', 'they', 'my', 'your', 'his', 'her', 'our', 'their', 'me', 'him', 'us', 'them', 'how', 'what', 'when', 'where', 'why', 'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did'}
    
    def generate_wordcloud(text, title):
        if not text.strip():
            st.warning("No text available for word cloud generation.")
            return None
            
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            stopwords=FILLER_WORDS,
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=20)
        return fig
    
    if wordcloud_option == "All Queries":
        all_text = ' '.join(df['query'].tolist())
        fig = generate_wordcloud(all_text, "Word Cloud - All Queries")
        if fig:
            st.pyplot(fig)
            
    elif wordcloud_option == "Current Selection":
        cluster_text = ' '.join(filtered_df['query'].tolist())
        fig = generate_wordcloud(cluster_text, f"Word Cloud - {selected_super_cluster}")
        if fig:
            st.pyplot(fig)
        
    elif wordcloud_option == "Selected Super Cluster":
        selected_for_wc = st.selectbox(
            "Select Super Cluster for word cloud:",
            df['super_cluster_name'].unique()
        )
        
        cluster_text = ' '.join(df[df['super_cluster_name'] == selected_for_wc]['query'].tolist())
        fig = generate_wordcloud(cluster_text, f"Word Cloud - {selected_for_wc}")
        if fig:
            st.pyplot(fig)
            
    elif wordcloud_option == "Compare Super Clusters" or wordcloud_option == "Compare with Another Cluster":
        st.subheader("Compare Word Clouds")
        
        col1, col2 = st.columns(2)
        
        cluster_names = df['super_cluster_name'].unique()
        
        with col1:
            if selected_super_cluster != 'All':
                st.write(f"**{selected_super_cluster}**")
                text1 = ' '.join(filtered_df['query'].tolist())
                fig1 = generate_wordcloud(text1, selected_super_cluster)
                if fig1:
                    st.pyplot(fig1)
            else:
                cluster1 = st.selectbox("First Super Cluster:", cluster_names, key="wc1")
                text1 = ' '.join(df[df['super_cluster_name'] == cluster1]['query'].tolist())
                fig1 = generate_wordcloud(text1, cluster1)
                if fig1:
                    st.pyplot(fig1)
        
        with col2:
            if selected_super_cluster != 'All':
                other_clusters = [c for c in cluster_names if c != selected_super_cluster]
                cluster2 = st.selectbox("Compare with:", other_clusters, key="wc2")
            else:
                cluster2 = st.selectbox("Second Super Cluster:", cluster_names, key="wc2")
            
            text2 = ' '.join(df[df['super_cluster_name'] == cluster2]['query'].tolist())
            fig2 = generate_wordcloud(text2, cluster2)
            if fig2:
                st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit | Query Clustering Analysis")