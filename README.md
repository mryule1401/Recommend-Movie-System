# Recommend-Movie-System

## Introduction
This project builds a movie recommender system that combines similarity-based techniques and KNN-based approaches to understand user preferences and suggest movies that are likely to be interesting. The system is implemented as an interactive Streamlit application so users can explore recommendations easily.

### Data
The project uses two main data sources:
- The MovieLens 100K dataset, which contains user ratings for movies and is used for collaborative filtering.
- TMDB movie metadata, which provides movie information such as title, overview, genres, and related attributes for content-based recommendations.

The data is stored in the project folders under the data directory, including the MovieLens 100K files and the TMDB movie CSV files.

### implement recommended algorith
The recommendation system uses several approaches:
- Content-based filtering: movie similarity is computed from textual metadata and tags using cosine similarity.
- Collaborative filtering: KNN-based methods are applied to user-item rating data to estimate preferences from similar users or similar items.
- Hybrid recommendation: the system combines content similarity and collaborative signals to produce more balanced movie suggestions.

In the web app, users can select a movie to receive similar recommendations or choose a user ID to view KNN-based predictions.

### libraries and technologies
The project is developed with the following tools and libraries:
- Python as the main programming language.
- NumPy and Pandas for data processing.
- scikit-learn for vectorization and similarity computation.
- Streamlit for building the interactive user interface.
- Requests for fetching movie poster images from TMDB.
- Pickle for loading pre-trained models and artifacts.

### performance evaluation
The system can be evaluated by measuring how relevant the recommended movies are for users. Common evaluation methods include:
- Precision@K and Recall@K for the top-N recommended items.
- Hit Rate to check whether relevant movies appear in the recommended list.
- Comparison between content-based, collaborative, and hybrid approaches to identify the most effective method.

These metrics help determine whether the recommender produces accurate and useful suggestions.

### development orientation
Future improvements can focus on:
- Expanding the dataset with more movies and richer metadata.
- Adding user feedback and personalization features.
- Improving the recommendation model with matrix factorization or deep learning techniques.
- Enhancing the UI with explanations for why each movie is recommended.

### conclusion
This project demonstrates how recommendation systems can be built using machine learning techniques to provide personalized movie suggestions. By combining content-based and collaborative filtering, the system offers a practical and effective approach to improving user experience in movie discovery.