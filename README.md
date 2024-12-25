# Movie Recommendation System 🎥🍿

This is a simple movie recommendation system that uses cosine similarity to recommend movies based on their genres, keywords, and overview. The system processes metadata from popular movie datasets and allows users to find movies similar to their favorite titles. It utilizes TF-IDF Vectorization and cosine similarity for computing recommendations. The system outputs the top 10 most similar movies, along with their genres.

Dataset limited to 5000 rows to enhance performance and reduce memory usage.
<br>

## Dataset

The required datasets are:

    movies_metadata.csv
    keywords.csv

These datasets are included in the Dataset directory. They are sourced from the [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) on Kaggle.

<br>

## Installation and Setup

  Clone the repository:

    git clone https://github.com/AliAbdullahSadiq/movie-recommendation-system.git
    cd movie-recommendation-system

Install the required Python packages:

    pip install -r requirements.txt

Ensure the Dataset directory contains the required CSV files (movies_metadata.csv and keywords.csv).

Run the script:

    python Movie_Recommendations.py

<br>

## Usage

  Modify the script to specify the movie title for which you want recommendations.
  Example: Get recommendations for Toy Story by editing:

    print(recommend_movies('Toy Story'))
