import streamlit as st
from recommender_engine import load_data, preprocess, MovieRecommender

# Page Configuration
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def get_engine():
    # Load dataset and initialize recommendation engine
    df = load_data("tmdb_5000_movies.csv")
    df = preprocess(df)
    return MovieRecommender(df), df


st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering with TF-IDF + cosine similarity")

# Initialize engine
engine, df = get_engine()

# Create Tabs for layout
tab1, tab2, tab3 = st.tabs(["🔎 Find Similar", "🎭 By Genre", "⭐ Personalized"])

# ---- Tab 1: Content-based similarity ---------------------------------------
with tab1:
    st.subheader("Movies similar to one you like")
    all_titles = sorted(df["title"].tolist())
    movie_choice = st.selectbox("Pick a movie", all_titles, key="single")
    n = st.slider("Number of recommendations", 5, 20, 10, key="n1")

    if st.button("Recommend", key="btn1"):
        results = engine.recommend(movie_choice, top_n=n)
        st.dataframe(results, use_container_width=True)

# ---- Tab 2: Genre-based ----------------------------------------------------
with tab2:
    st.subheader("Top-rated movies by genre")
    genres = ["Action", "Comedy", "Drama", "Horror", "Science Fiction",
              "Romance", "Thriller", "Animation", "Adventure", "Fantasy"]
    genre_choice = st.selectbox("Pick a genre", genres)
    n2 = st.slider("Number of recommendations", 5, 20, 10, key="n2")

    if st.button("Recommend", key="btn2"):
        results = engine.recommend_by_genre(genre_choice, top_n=n2)
        st.dataframe(results, use_container_width=True)

# ---- Tab 3: Personalized ----------------------------------------------------
with tab3:
    st.subheader("Recommendations based on movies you've liked")
    all_titles = sorted(df["title"].tolist())
    liked = st.multiselect("Select a few movies you enjoyed", all_titles)
    n3 = st.slider("Number of recommendations", 5, 20, 10, key="n3")

    if st.button("Recommend", key="btn3"):
        if not liked:
            st.warning("Pick at least one movie first.")
        else:
            results = engine.recommend_for_user(liked, top_n=n3)
            st.dataframe(results, use_container_width=True)

st.divider()
st.caption(f"Dataset: {len(df)} movies · TMDB 5000 Movie Metadata")