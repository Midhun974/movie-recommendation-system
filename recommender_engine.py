import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(filepath):
    """Loads the TMDB 5000 movies CSV file."""
    df = pd.read_csv(filepath)
    return df


def preprocess(df):
    """Cleans overview text and extracts genres."""

    def extract_genres(genre_str):
        try:
            genres = json.loads(genre_str)
            return " ".join([g["name"] for g in genres])
        except:
            return ""

    # Ensure overview and genres are string values
    df["overview"] = df["overview"].fillna("")
    if "genres" in df.columns:
        df["genre_names"] = df["genres"].apply(extract_genres)
    else:
        df["genre_names"] = ""

    # Combine text features for TF-IDF
    df["combined_features"] = df["overview"] + " " + df["genre_names"]
    return df


class MovieRecommender:

    def __init__(self, df):
        self.df = df
        # Create TF-IDF matrix
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(self.df["combined_features"])
        # Calculate Cosine Similarity
        self.cosine_sim = cosine_similarity(
            self.tfidf_matrix, self.tfidf_matrix
        )
        self.indices = pd.Series(
            self.df.index, index=self.df["title"]
        ).drop_duplicates()

    def recommend(self, title, top_n=10):
        """Recommends similar movies based on content similarity."""
        if title not in self.indices:
            return pd.DataFrame(columns=["title", "vote_average"])

        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[
            1 : top_n + 1
        ]

        movie_indices = [i[0] for i in sim_scores]
        return self.df[["title", "vote_average"]].iloc[movie_indices]

    def recommend_by_genre(self, genre, top_n=10):
        """Filters top-rated movies matching a specific genre."""
        genre_df = self.df[
            self.df["genre_names"].str.contains(genre, case=False, na=False)
        ]
        sorted_df = genre_df.sort_values(by="vote_average", ascending=False)
        return sorted_df[["title", "vote_average"]].head(top_n)

    def recommend_for_user(self, liked_titles, top_n=10):
        """Recommends movies based on multiple user-liked movies."""
        valid_titles = [t for t in liked_titles if t in self.indices]
        if not valid_titles:
            return pd.DataFrame(columns=["title", "vote_average"])

        # Average similarity vectors across all liked movies
        liked_indices = [self.indices[t] for t in valid_titles]
        avg_sim = np.mean([self.cosine_sim[idx] for idx in liked_indices], axis=0)

        sim_scores = list(enumerate(avg_sim))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude already liked movies
        sim_scores = [
            (i, score)
            for i, score in sim_scores
            if self.df.iloc[i]["title"] not in liked_titles
        ]

        movie_indices = [i[0] for i in sim_scores[:top_n]]
        return self.df[["title", "vote_average"]].iloc[movie_indices]