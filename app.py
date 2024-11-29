import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tensorflow import keras
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util
import pickle
import torch

# Load CSS for background image and navigation bar
st.markdown(
    """
    <style>
    .main {
        background-image: url("https://tse4.mm.bing.net/th?id=OIP.QOKLFI7-SMhiupT6OwdiWwHaDe&pid=Api&P=0&h=180");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# create navigation bar
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to",
                        ['Home', 'Recommendation (Sentence Transformers)', 'Recommendation (TF-IDF)', 'Prediction',
                         'Trending'])

# Load data
arxiv_data = pd.read_csv("arxiv_data_210930-054931.csv")
arxiv_data.drop_duplicates(subset=['titles'], inplace=True)
arxiv_data.reset_index(drop=True, inplace=True)

def invert_multi_hot(encoded_labels):
    """Reverse a single multi-hot encoded label to a tuple of vocab terms."""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)


# Load the model for subject area prediction
loaded_model = keras.models.load_model("models/model.h5")
# Load the configuration of the text vectorizer
with open("models/text_vectorizer_config.pkl", "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)
# Create a new TextVectorization layer with the saved configuration
loaded_text_vectorizer = layers.TextVectorization.from_config(saved_text_vectorizer_config)
# Load the saved weights into the new TextVectorization layer
with open("models/text_vectorizer_weights.pkl", "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)
# Load the vocabulary
with open("models/vocab.pkl", "rb") as f:
    loaded_vocab = pickle.load(f)

# Sample trending topics
trending_topics = ["Machine Learning", "Natural Language Processing", "Computer Vision", "Deep Learning",
                   "Data Science"]

# Load saved recommendation models
embeddings = pickle.load(open('models/embeddings.pkl', 'rb'))
sentences = pickle.load(open('models/sentences.pkl', 'rb'))
rec_model = pickle.load(open('models/rec_model.pkl', 'rb'))

# Function for recommendation using Sentence Transformers
def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list


# Function to visualize recommendations
def visualize_recommendations(recommend_papers):
    # Plot keyword frequencies for the recommended papers
    fig, ax = plt.subplots(figsize=(10, 6))
    keywords = ["machine learning", "deep learning", "natural language processing", "computer vision", "data science"]
    keyword_counts = {keyword: sum(1 for paper in recommend_papers if keyword in paper.lower()) for keyword in keywords}
    sns.barplot(x=list(keyword_counts.keys()), y=list(keyword_counts.values()), ax=ax)
    ax.set_title('Keyword Frequencies in Recommended Papers')
    ax.set_xlabel('Keyword')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)


# Function for recommendation using TF-IDF and KNN
def recommendation_tfidf_knn(input_title, input_abstract):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=3)
    tfidf_matrix = tfidf_vectorizer.fit_transform(arxiv_data['titles'] + ' ' + arxiv_data['abstracts'])
    knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn_model.fit(tfidf_matrix)
    input_vector = tfidf_vectorizer.transform([input_title + ' ' + input_abstract])
    distances, indices = knn_model.kneighbors(input_vector)
    return arxiv_data.iloc[indices[0]]['titles'].tolist()


# Function for subject area prediction
def predict_category(abstract, model, vectorizer, label_lookup):
    # Preprocess the abstract using the loaded text vectorizer
    preprocessed_abstract = vectorizer([abstract])

    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_abstract)

    # Convert predictions to human-readable labels
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])

    return predicted_labels


# Function to extract trending topics using Latent Dirichlet Allocation (LDA)
def extract_trending_topics(data, num_topics=5, num_top_words=5):
    # Vectorize the text data
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=3)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['titles'] + ' ' + data['abstracts'])

    # Apply LDA to extract topics
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)

    # Get the top words for each topic
    topic_keywords = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_top_words - 1:-1]
        topic_keywords.append([tfidf_vectorizer.get_feature_names_out()[i] for i in top_words_idx])

    return topic_keywords


# create app=========================================
# Home Page
# Home Page
if page == 'Home':
    st.title('Welcome to our Research Papers Recommendation App')
    st.write(
        "Welcome to our Research Papers Recommendation App! This app helps you discover relevant research papers, predict subject areas, explore trending topics, and more.")

    # Introduction Section
    st.header("About the App")
    st.markdown("""
    Research papers are crucial resources for staying updated with the latest advancements in various fields. However, with the vast amount of research being published every day, finding relevant papers can be challenging.

    Our Research Papers Recommendation App leverages machine learning models and natural language processing techniques to help you discover research papers tailored to your interests and needs.
    """)

    # Features Section
    st.header("Key Features")
    st.markdown("""
    - **Recommendation:** Get personalized recommendations for research papers based on your input title and abstract.
    - **Subject Area Prediction:** Predict the subject area of a research paper based on its abstract.
    - **Trending Topics:** Explore trending topics in research fields using Latent Dirichlet Allocation.
    - **Search:** Search for specific research papers based on keywords or queries.
    """)

    # Recent Highlights Section
    st.header("Recent Highlights")
    st.markdown("""
    - Improved recommendation algorithms for better accuracy.
    - Enhanced user interface for a more intuitive experience.
    - Updated dataset with the latest research papers.
    """)

    # Quick Navigation Section
    st.header("Quick Navigation")
    st.markdown("""
    - **Recommendation (Sentence Transformers):** Discover research papers using Sentence Transformers.
    - **Recommendation (TF-IDF):** Explore research papers using TF-IDF and KNN.
    - **Prediction:** Predict the subject area of a research paper.
    - **Trending:** Explore trending topics in research.
    - **Search:** Search for specific research papers.
    """)


elif page == 'Recommendation (Sentence Transformers)':
    st.title('Research Papers Recommendation (Sentence Transformers)')
    input_paper = st.text_input("Enter Paper title.....")
    if st.button("Recommend"):
        # recommendation part
        recommend_papers = recommendation(input_paper)
        st.subheader("Recommended Papers")
        st.write(recommend_papers)

        # Visualize recommended papers
        visualize_recommendations(recommend_papers)

elif page == 'Recommendation (TF-IDF)':
    st.title('Research Papers Recommendation (TF-IDF)')
    input_title = st.text_input("Enter Paper title.....")
    input_abstract = st.text_area("Enter Abstract")
    if st.button("Recommend"):
        # recommendation using TF-IDF and KNN
        recommend_papers = recommendation_tfidf_knn(input_title, input_abstract)
        st.subheader("Recommended Papers (TF-IDF and KNN)")
        st.write(recommend_papers)

elif page == 'Prediction':
    st.title('Research Papers Subject Area Prediction')
    new_abstract = st.text_area("Enter Abstract")
    if st.button("Predict", key="predict_button"):
        st.markdown(
            """
            <style>
            #predict_button {
                background-color: red !important;
                color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # ========prediction part
        predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)
        st.subheader("Predicted Subject area")
        st.write(predicted_categories)

elif page == 'Trending':
    st.title('Trending Topics')
    st.write("Check out the trending topics in research:")
    st.write(trending_topics)

    # Extract trending topics using LDA
    num_topics = st.slider("Number of Topics", min_value=2, max_value=10, value=5)
    trending_topics = extract_trending_topics(arxiv_data, num_topics)
    st.subheader('Extracted Trending Topics')
    for i, topic in enumerate(trending_topics):
        st.write(f"Topic {i + 1}: {', '.join(topic)}")
