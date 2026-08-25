# 🎬 Movie Recommender

A movie recommendation system built with Python and Streamlit.

The application allows users to search for movies and receive recommendations for movies that are similar to the selected movie.

## ✨ Features

- 🔎 Search for movies by title
- 🎯 Recommend similar movies
- ⭐ Display movie ratings
- 🗳️ Display vote counts
- 📅 Display release dates
- 🖼️ Display movie posters
- 📖 Display movie descriptions
- 🌙 Dark mode and ☀️ Light mode
- ⚡ Fast similarity search using NearestNeighbors
- 💾 Efficient loading with Streamlit caching
- 📦 Parquet dataset for faster data loading

## 🛠️ Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Joblib
- PyArrow

## 🧠 Recommendation System

The recommendation system uses TF-IDF to represent movie information as numerical vectors.

NearestNeighbors with cosine distance is then used to find movies with similar feature vectors.

The application returns the most similar movies based on their similarity scores.

## 📂 Project Structure

```text
movie-recommender/
│
├── data/
│   └── movies.parquet
│
├── notebook/
│   ├── 01_data_exploration.ipynb
│   └── 02_movie_recommender_new.ipynb
│
├── app.py
├── build_faiss.py
├── convert_dataset.py
├── convert_tfidf.py
├── requirements.txt
├── tfidf_shape.txt
├── .gitignore
└── README.md
