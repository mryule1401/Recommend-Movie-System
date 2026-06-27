import pickle
import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define KNN class to support unpickling saved models
class KNearestNeighbor:
    def __init__(self, k, n_items, n_user, uuCF=1, dist_f=cosine_similarity, limit=10):
        self.k = k
        self.n_items = n_items
        self.n_user = n_user
        self.uuCF = uuCF
        self.dist_f = dist_f
        self.limit = limit
        self.user_similarity = None
        self.item_similarity = None
        self.Ybar = None
        self.mu = None

    def fit(self, Y_data):
        if self.uuCF == 1:
            self.user_similarity = self.dist_f(Y_data.T, Y_data.T)
            self.Ybar = Y_data.copy()
        else:
            self.item_similarity = self.dist_f(Y_data, Y_data)
            self.Ybar = Y_data.copy()
            self.mu = np.mean(Y_data, axis=1)
            for i in range(self.n_items):
                ids = np.where(Y_data[i] != 0)[0]
                self.Ybar[i, ids] -= self.mu[i]

    def predict_one_user(self, u, Y_data):
        similarities = self.user_similarity[u]
        ids = np.argsort(similarities)[-self.k:]
        nearest_neighbor_ratings = Y_data[ids, :]
        sigma_similarities = np.sum(np.abs(similarities[ids]))
        return np.dot(similarities[ids], nearest_neighbor_ratings) / (sigma_similarities + 1e-8)

    def predict_one_item(self, i, Y_data):
        similarities = self.item_similarity[i]
        ids = np.argsort(similarities)[-self.k:]
        nearest_neighbor_ratings = Y_data[ids, :]
        sigma_similarities = np.sum(np.abs(similarities[ids]))
        rating_bar = (self.Ybar[ids, :].T / np.abs(similarities[ids])).T
        return self.mu[i] + np.dot(similarities[ids], rating_bar) / (sigma_similarities + 1e-8)

    def predict(self, Y_data):
        if self.uuCF == 1:
            predictions = np.zeros((Y_data.shape[0], Y_data.shape[1]))
            for u in range(self.n_user):
                predictions[:, u] = self.predict_one_user(u, Y_data)
        else:
            predictions = np.zeros((self.n_items, self.n_user))
            for i in range(self.n_items):
                predictions[i, :] = self.predict_one_item(i, Y_data)
        return predictions

@st.cache_resource
def load_models():
    """Load all pre-trained models from artifacts"""
    movies = pickle.load(open('artifacts/movie_list.pkl','rb'))
    similarity = pickle.load(open('artifacts/similarity.pkl','rb'))
    knn_uu = pickle.load(open('artifacts/knn_user_user.pkl','rb'))
    knn_ii = pickle.load(open('artifacts/knn_item_item.pkl','rb'))
    
    # Load ML-100k data
    ratings = pd.read_csv(
        'data/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        encoding='latin-1'
    )
    items = pd.read_csv(
        'data/ml-100k/u.item',
        sep='|',
        names=['item_id', 'title'] + [f'f{i}' for i in range(22)],
        usecols=[0, 1],
        encoding='latin-1',
        engine='python'
    )
    item_mapping = dict(zip(items['item_id'], items['title']))
    
    # Build rating matrix for KNN predictions
    n_users = int(ratings['user_id'].max())
    n_items = int(ratings['item_id'].max())
    rating_matrix = np.zeros((n_items, n_users), dtype=np.float32)
    for row in ratings.itertuples(index=False):
        rating_matrix[row.item_id - 1, row.user_id - 1] = row.rating
    item_similarity = cosine_similarity(rating_matrix)
    
    return {
        'movies': movies,
        'similarity': similarity,
        'knn_uu': knn_uu,
        'knn_ii': knn_ii,
        'item_mapping': item_mapping,
        'rating_matrix': rating_matrix,
        'item_similarity': item_similarity
    }

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie, mode='Similarity', movies=None, similarity=None, tag_matrix=None):
    index = movies[movies['title'] == movie].index[0]
    if mode == 'Hybrid content':
        content_scores = cosine_similarity(tag_matrix[index], tag_matrix).flatten()
        scores = (similarity[index] + content_scores) / 2
    else:
        scores = similarity[index]

    distances = sorted(list(enumerate(scores)), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


def knn_user_recommendations(user_id=1, top=5, knn_model=None, rating_matrix=None, item_similarity=None, item_mapping=None, k=30):
    user_index = user_id - 1
    user_ratings = rating_matrix[:, user_index]
    unrated_items = np.where(user_ratings == 0)[0]
    predictions = []

    for item_idx in unrated_items:
        sim_scores = item_similarity[item_idx].copy()
        sim_scores[item_idx] = 0
        top_neighbors = np.argsort(sim_scores)[-k:]
        neighbor_ratings = user_ratings[top_neighbors]
        neighbor_sims = sim_scores[top_neighbors]
        rated_mask = neighbor_ratings > 0
        if not np.any(rated_mask):
            continue
        score = np.dot(neighbor_sims[rated_mask], neighbor_ratings[rated_mask])
        score = score / np.sum(np.abs(neighbor_sims[rated_mask]))
        predictions.append((item_idx, score))

    predictions.sort(key=lambda x: x[1], reverse=True)
    recommended_titles = [item_mapping[item_idx + 1] for item_idx, _ in predictions[:top]]
    return recommended_titles


st.header('Movie Recommender System Using Machine Learning')

# Load all models
models = load_models()
movies = models['movies']
similarity = models['similarity']
knn_uu = models['knn_uu']
knn_ii = models['knn_ii']
item_mapping = models['item_mapping']
rating_matrix = models['rating_matrix']
item_similarity = models['item_similarity']

vectorizer = CountVectorizer(max_features=5000, stop_words='english')
tag_matrix = vectorizer.fit_transform(movies['tags'])

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

recommendation_type = st.radio(
    "Recommendation mode",
    ('Similarity', 'KNN recommended')
)

selected_user = 1
if recommendation_type == 'KNN recommended':
    selected_user = st.selectbox(
        "Select user for KNN recommendations",
        list(range(1, 21)),
        index=0
    )
    st.info(f'KNN mode uses ML-100k user behavior data and shows top 5 recommendations for user {selected_user}.')

if st.button('Show Recommendation'):
    if recommendation_type == 'KNN recommended':
        recommended_titles = knn_user_recommendations(
            user_id=selected_user, 
            top=5,
            knn_model=knn_ii,
            rating_matrix=rating_matrix,
            item_similarity=item_similarity,
            item_mapping=item_mapping,
            k=30
        )
        st.markdown(f'### Top 5 KNN recommendations for user {selected_user} (ML-100k behavior data)')
        for title in recommended_titles:
            st.write('- ' + title)
    else:
        recommended_movie_names, recommended_movie_posters = recommend(
            selected_movie, 
            recommendation_type,
            movies=movies,
            similarity=similarity,
            tag_matrix=tag_matrix
        )
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])

        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])
