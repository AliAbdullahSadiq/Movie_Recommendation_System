import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pathlib import Path

base_dir = Path(__file__).resolve().parent

movies_metadata_path = base_dir / 'Dataset' / 'movies_metadata.csv'
keywords_path = base_dir / 'Dataset' / 'keywords.csv'

# Load the datasets
movies = pd.read_csv(movies_metadata_path, low_memory=False)
keywords = pd.read_csv(keywords_path)

# Preprocess: Select important columns
movies = movies[['id', 'title', 'overview', 'genres']]

# Fix data type of 'id' column in both datasets
movies['id'] = movies['id'].astype(str)
keywords['id'] = keywords['id'].astype(str)

# Merge keywords into movies dataset
movies = movies.merge(keywords, on='id', how='left')

# Ensure 'keywords' column contains lists or empty lists
movies['keywords'] = movies['keywords'].apply(lambda x: x if isinstance(x, list) else [])

# Preprocess genres and combine features
movies['genres'] = movies['genres'].fillna('[]').apply(eval).apply(lambda x: [i['name'] for i in x])
movies['combined_features'] = movies['genres'].apply(lambda x: ' '.join(x)) + ' ' + \
                              movies['keywords'].apply(lambda x: ' '.join(x)) + ' ' + \
                              movies['overview'].fillna('')

# Limit dataset size for testing
movies = movies.head(5000)  # Use the first 5000 rows

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

# Compute cosine similarity using linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend_movies(title, cosine_sim=cosine_sim):
    # Check if the movie title exists in the dataset
    if title not in movies['title'].values:
        return f"Movie '{title}' not found in the dataset."

    # Get the index of the movie that matches the title
    idx = movies[movies['title'] == title].index[0]

    # Get the pairwise similarity scores for that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies.iloc[movie_indices][['title', 'genres']]


print(recommend_movies('The Godfather'))