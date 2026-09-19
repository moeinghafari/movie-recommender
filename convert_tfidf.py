import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer

print("در حال بارگذاری دیتاست...")

start = time.time()

movies = pd.read_parquet("data/movies.parquet")

print(
    f"دیتاست بارگذاری شد: "
    f"{time.time() - start:.2f} ثانیه"
)

print("تعداد فیلم‌ها:", len(movies))


movies["tags"] = (
    movies["tags"]
    .fillna("")
    .astype(str)
)


print("در حال ساخت TF-IDF...")

start = time.time()

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    dtype=np.float32,
    sublinear_tf=True
)

tfidf_matrix = tfidf.fit_transform(
    movies["tags"]
)

print(
    f"TF-IDF ساخته شد: "
    f"{time.time() - start:.2f} ثانیه"
)

print("Shape:", tfidf_matrix.shape)


print("در حال ذخیره‌سازی...")


np.save(
    "tfidf_data.npy",
    tfidf_matrix.data
)

np.save(
    "tfidf_indices.npy",
    tfidf_matrix.indices
)

np.save(
    "tfidf_indptr.npy",
    tfidf_matrix.indptr
)


with open("tfidf_shape.txt", "w") as f:
    f.write(
        f"{tfidf_matrix.shape[0]},"
        f"{tfidf_matrix.shape[1]}"
    )


print("تبدیل با موفقیت انجام شد.")